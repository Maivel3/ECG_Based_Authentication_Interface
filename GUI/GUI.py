import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
import scipy
import pandas as pd
from tkinter import PhotoImage
import re
import os
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt, ellip, decimate
import statsmodels.api as sm
from scipy import signal
from sklearn.datasets import load_iris
from tkinter import Tk, Label

from PIL import ImageTk, Image
class ECGClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.file_path = ""
        self.signal = []
        window_width = 800
        window_height = 600
        root.geometry(f"{window_width}x{window_height}")

        # Load the background image
        image = Image.open("C:\\Users\\dell\\Downloads\\7244.png")
        image = image.resize((window_width, window_height), Image.ANTIALIAS)
        self.background_image = ImageTk.PhotoImage(image)

        # Create a label to hold the background image
        background_label = tk.Label(root, image=self.background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.select_button = tk.Button(self.root, text="Select ECG File", width=15, height=2, font=("Arial", 16),
                                       command=self.select_file)
        self.select_button.pack(side='left', padx=10, pady=10)
        self.select_button.place(x=5,y=250)

        self.draw_button = tk.Button(self.root, text="Draw Signal", width=15, height=2, font=("Arial", 16),
                                     command=self.draw_signal, state=tk.NORMAL)
        self.draw_button.pack(side='left', padx=10, pady=10)
        self.draw_button.place(x=200, y=250)
        self.classify_button = tk.Button(self.root, text="Check", width=15, height=2, font=("Arial", 16),
                                         command=self.classify_signal, state=tk.NORMAL)
        self.classify_button.pack(side='left', padx=10, pady=10)
        self.classify_button.place(x=400, y=250)
        self.Fiducial = tk.Button(self.root, text="Fiducial", width=15, height=2, font=("Arial", 16),
                                         command=self.feducial_Feature_Extraction, state=tk.NORMAL)
        self.Fiducial.pack(side='left', padx=10, pady=10)
        self.Fiducial.place(x=600, y=250)
        self.result_label = tk.Label(self.root, text="Class", width=15, height=2, font=("Arial", 16))
        self.result_label.pack(side='bottom', padx=10, pady=10)
        self.result_label.place(x=300,y=350)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if self.file_path:
            self.signal = np.loadtxt(self.file_path)
            self.draw_button["state"] = tk.NORMAL

    def draw_signal(self):
        plt.figure()
        plt.plot(self.signal)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.title("ECG Signal")
        plt.show()

    def detect_t_peaks(self,ecg_signal, r_peaks, sampling_rate):
        t_peaks = []
        final_peaks = []
        search_window_duration = int(0.4 * sampling_rate)
        wing_width = int(0.05 * sampling_rate)

        for r_peak in r_peaks:
            search_window_start = r_peak + int(0.04 * sampling_rate)

            if search_window_start >= len(ecg_signal):
                break

            search_window_end = search_window_start + search_window_duration
            search_window = ecg_signal[search_window_start:search_window_end]

            t_peaks.clear()

            for i in range(len(search_window)):
                if i >= wing_width and i < len(search_window) - wing_width:
                    w1 = search_window[i - wing_width:i]
                    w2 = search_window[i:i + wing_width]
                    wings_diff = np.sum(w2) - np.sum(w1)
                    if wings_diff < 0:  # Negative wing peak detected
                        t_peak_index = search_window_start + i
                        t_peaks.append(t_peak_index)

            if len(t_peaks) > 0:
                final_peaks.append(np.min(t_peaks))

        return final_peaks

    def detect_q_s_peaks(self,ecg_signal, r_peaks):
        q_peaks = []
        s_peaks = []

        for r_peak in r_peaks:

            q_peak = np.argmin(ecg_signal[max(0, r_peak - 50):r_peak]) + max(0, r_peak - 50)
            q_peaks.append(q_peak)

            if r_peak > 0:
                s_peak = np.argmin(ecg_signal[r_peak:min(r_peak + 100, len(ecg_signal))]) + r_peak
                s_peaks.append(s_peak)

        return q_peaks, s_peaks

    def detect_Tonset_Toffset(self,ecg_signal, r_peaks):
        q_peaks = []
        s_peaks = []

        for r_peak in r_peaks:
            q_peak = np.argmin(ecg_signal[max(0, r_peak - 50):r_peak]) + max(0, r_peak - 100)
            q_peaks.append(q_peak)
            s_peak = np.argmin(ecg_signal[r_peak:min(r_peak + 90, len(ecg_signal))]) + r_peak
            s_peaks.append(s_peak)

        return q_peaks, s_peaks

    def detect_qrs_onset_offset(self,ecg_signal, q_peaks, s_peaks):
        qrs_onsets = []
        qrs_offsets = []

        for q_peak, s_peak in zip(q_peaks, s_peaks):
            qrs_onset = 0
            qrs_offset = len(ecg_signal) - 1

            # Detect QRS onset (backward search from Q peak)
            for i in range(q_peak, 0, -1):
                if ecg_signal[i] > ecg_signal[i - 1]:
                    qrs_onset = i
                    break

            # Detect QRS offset (forward search from S peak)
            for i in range(s_peak, len(ecg_signal) - 1):
                if ecg_signal[i] > ecg_signal[i + 1]:
                    qrs_offset = i
                    break

            qrs_onsets.append(qrs_onset)
            qrs_offsets.append(qrs_offset)

        return qrs_onsets, qrs_offsets

    def detect_p_peak(self,ecg_signal, qrs_onsets, sampling_rate):
        p_peaks = []
        search_window_duration = 0.2
        ratio_threshold = 0.5
        duration_threshold = 0.08

        for qrs_onset in qrs_onsets:
            search_window_start = qrs_onset - int(0.2 * sampling_rate)
            search_window_end = search_window_start + int(search_window_duration * sampling_rate)

            search_window = ecg_signal[search_window_start:search_window_end]
            peak_indices, _ = find_peaks(search_window, distance=int(0.5 * sampling_rate))

            if len(peak_indices) == 1:
                p_peak = peak_indices[0] + search_window_start
            elif len(peak_indices) == 2:
                peak1 = peak_indices[0] + search_window_start
                peak2 = peak_indices[1] + search_window_start

                amplitude_ratio = ecg_signal[peak1] / ecg_signal[peak2]
                duration = peak2 - peak1

                if amplitude_ratio < ratio_threshold and duration < duration_threshold:
                    p_peak = peak2
                else:
                    p_peak = peak1
            else:
                p_peak = -1

            p_peaks.append(p_peak)

        return p_peaks

    def extract_fiducial_features(self,ecg_signal, sampling_rate):

        r_peaks, _ = find_peaks(ecg_signal, distance=int(0.5 * sampling_rate))

        t_peaks = self.detect_t_peaks(ecg_signal, r_peaks, sampling_rate)
        q_peaks, s_peaks = self.detect_q_s_peaks(ecg_signal, r_peaks)

        qrs_onsets, qrs_offsets = self.detect_qrs_onset_offset(ecg_signal, q_peaks, s_peaks)
        p_peaks = self.detect_p_peak(ecg_signal, qrs_onsets, sampling_rate)
        Ponset, Poffset = self.detect_q_s_peaks(ecg_signal, p_peaks)
        Tonset, Toffset = self.detect_Tonset_Toffset(ecg_signal, t_peaks)

        # Visualize the ECG signal with R-peaks and T-peaks marked
        plt.plot(ecg_signal)
        plt.plot(r_peaks, ecg_signal[r_peaks], 'bo', color='green', label='R-peaks')
        plt.plot(q_peaks, ecg_signal[q_peaks], 'bo', color='green', label='Q-peaks')
        plt.plot(s_peaks, ecg_signal[s_peaks], 'bo', color='green', label='S-peaks')
        plt.plot(qrs_onsets, ecg_signal[qrs_onsets], 'bo', color='g', label='QRSon')
        plt.plot(qrs_offsets, ecg_signal[qrs_offsets], 'bo', color='g', label='QRSoff')
        plt.plot(p_peaks, ecg_signal[p_peaks], 'bo', c='r', label='P-peaks')
        plt.plot(Ponset, ecg_signal[Ponset], 'bo', color='r', label='Ponset')
        plt.plot(Poffset, ecg_signal[Poffset], 'bo', color='r', label='Poffset')
        plt.plot(t_peaks, ecg_signal[t_peaks], 'bo', c='b', label='T-peaks')
        plt.plot(Tonset, ecg_signal[Tonset], 'bo', color='b', label='Tonset')
        plt.plot(Toffset, ecg_signal[Toffset], 'bo', color='b', label='Toffset')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('ECG Signal with R-peaks and T-peaks')
        plt.legend()
        plt.grid(True)
        plt.show()

        return r_peaks, q_peaks, s_peaks, qrs_onsets, qrs_offsets, p_peaks, Ponset, Poffset, t_peaks, Tonset, Toffset
    def feducial_Feature_Extraction(self):
        if self.signal.ndim > 1:
            signal = self.signal.ravel()

        signal_dc_removed = self.signal - np.mean(self.signal)
        lowcut = 0.5
        highcut = 40
        rp = 0.1
        rs = 60
        fs = 1000
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = ellip(3, rp, rs, [low, high], btype='band')

        signal_filtered = filtfilt(b, a, signal_dc_removed)
        signal_normalized = (signal_filtered - np.mean(signal_filtered)) / np.std(signal_filtered)
        features_dict = {
            'R Peaks': [],
            'Q Peaks': [],
            'S Peaks': [],
            'QRS Onsets': [],
            'QRS Offsets': [],
            'P Peaks': [],
            'P Onset': [],
            'P Offset': [],
            'T Peaks': [],
            'T Onset': [],
            'T Offset': []

        }
        # Extract fiducial features
        r_peaks, q_peaks, s_peaks, qrs_onsets, qrs_offsets, p_peaks, Ponset, Poffset, t_peaks, Tonset, Toffset = self.extract_fiducial_features(
            signal_normalized, 1000)

        for peak in r_peaks:
            features_dict['R Peaks'].append(peak)
        for peak in q_peaks:
            features_dict['Q Peaks'].append(peak)
        for peak in s_peaks:
            features_dict['S Peaks'].append(peak)
        for peak in qrs_onsets:
            features_dict['QRS Onsets'].append(peak)
        for peak in qrs_offsets:
            features_dict['QRS Offsets'].append(peak)
        for peak in p_peaks:
            features_dict['P Peaks'].append(peak)
        for peak in Ponset:
            features_dict['P Onset'].append(peak)
        for peak in Poffset:
            features_dict['P Offset'].append(peak)
        for peak in t_peaks:
            features_dict['T Peaks'].append(peak)
        for peak in Tonset:
            features_dict['T Onset'].append(peak)
        for peak in Toffset:
            features_dict['T Offset'].append(peak)
        max_length = max(len(arr) for arr in features_dict.values())
        padded_data = {key: arr + [None] * (max_length - len(arr)) for key, arr in features_dict.items()}

        df = pd.DataFrame(padded_data)
        df = df.dropna()
        X=df
        np.array(X)


        model = joblib.load("fiducial_svm_model (1).pkl")
        class_label = model.predict( np.array(X))
        y_prob = model.predict_proba( np.array(X))
        x=y_prob.max()
        print(x)
        if x< 0.35:
            self.result_label.config(text="Not Authenticated")
        else:
            self.result_label.config(text=f"Authenticated!! \n {class_label}")


    def classify_signal(self):

        if self.signal.ndim > 1:
            signal = self.signal.ravel()

        signal_dc_removed = self.signal - np.mean(self.signal)
        lowcut = 0.5
        highcut = 40
        rp = 0.1
        rs = 60
        fs=1000
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = ellip(3, rp, rs, [low, high], btype='band')

        signal_filtered = filtfilt(b, a, signal_dc_removed)
        signal_normalized = (signal_filtered - np.mean(signal_filtered)) / np.std(signal_filtered)


        AC1 = sm.tsa.acf(signal_normalized, nlags=5000)
        s1 = AC1[0:500]
        DCT_1 = scipy.fftpack.dct(s1, type=2)

        model = joblib.load("svm_model25.pkl")
        class_label = model.predict([DCT_1])[0]

        data_2d = DCT_1.reshape(1, -1)


        y_prob = model.predict_proba(data_2d)
        x=y_prob[0].max()
        print(x)
        if x< 0.6:
            self.result_label.config(text="Not Authenticated")
        else:
            self.result_label.config(text=f"Authenticated!! \n {class_label}")

root = tk.Tk()

app = ECGClassificationGUI(root)



root.mainloop()
