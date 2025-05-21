import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pywt
import pandas as pd
from scipy.signal import find_peaks

mat        = scipy.io.loadmat('JS03454.mat')
raw        = mat['val'] 
sig2d      = raw.squeeze() 
if sig2d.ndim > 1:
    signal = sig2d[0, :].astype(np.float64)
else:
    signal = sig2d.astype(np.float64)

fs = 500
t  = np.arange(len(signal)) / fs

def wavelet_denoising(data, wavelet='db4', level=8, threshold_mode='soft'):
    coeffs    = pywt.wavedec(data, wavelet, level=level)
    sigma     = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh    = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeff = [coeffs[0]]
    for c in coeffs[1:]:
        new_coeff.append(pywt.threshold(c, thresh, mode=threshold_mode))
    rec = pywt.waverec(new_coeff, wavelet)
    return rec[:len(data)]

filtsignal = wavelet_denoising(signal)

class Symlet4_QRS:
    def __init__(self, fs=500, level=3, thr_factor=0.5, smooth_sec=0.15):
        self.fs         = fs
        self.level      = level
        self.thr_factor = thr_factor
        # window in samples
        self.win        = max(1, int(smooth_sec * fs))

    def solve(self, sig):
        coeffs = pywt.wavedec(sig, 'sym4', level=self.level)
        # zero all subbands except detail at self.level-1
        zeros  = [np.zeros_like(c) for c in coeffs]
        zeros[self.level-1] = coeffs[self.level-1]
        detail = pywt.waverec(zeros, 'sym4')[:len(sig)]
        env    = np.abs(detail).astype(np.float64)
        env    = np.convolve(env, np.ones(self.win)/self.win, mode='same')
        thr    = self.thr_factor * np.max(env)
        dist   = int(0.2 * self.fs)
        peaks, _ = find_peaks(env, height=thr, distance=dist)
        return env, peaks

detector = Symlet4_QRS(fs=fs, level=3, thr_factor=0.5)
env, peaks = detector.solve(filtsignal)

r_times     = peaks / fs
rr_intervals = np.diff(r_times)
hr          = 60.0 / rr_intervals
hr_times    = r_times[1:]




plt.figure(figsize=(10,4))
plt.plot(t, signal)
plt.title('Original ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig("Original_ECG_Signal.png")

plt.figure(figsize=(10,4))
plt.plot(t, filtsignal)
plt.title('Filtered ECG Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.savefig("Filtered_ECG_Signal.png")

plt.figure(figsize=(10,4))
plt.plot(t, signal,  label='Original')
plt.plot(t, filtsignal, label='Filtered')
plt.title('Original vs Filtered ECG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Original_vs_Filtered_ECG.png")

plt.figure(figsize=(10,4))
plt.plot(t, env, label='Envelope')
plt.scatter(r_times, env[peaks],
            s=60, facecolors='none', edgecolors='r', lw=1.5,
            label='QRS Complexes')
plt.title('Wavelet Envelope with QRS Complexes')
plt.xlabel('Time (s)')
plt.ylabel('Envelope')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("QRS_Complexes.png")

plt.figure(figsize=(10,4))
plt.plot(t, filtsignal, label='Filtered ECG')
plt.scatter(r_times, filtsignal[peaks],
            s=60, facecolors='none', edgecolors='m', lw=1.5,
            label='R-peaks')
plt.title('R-peaks on Filtered ECG')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("R_peaks.png")

plt.figure(figsize=(10,4))
plt.plot(hr_times, rr_intervals, 'go-')
plt.title('RR Intervals vs Time')
plt.xlabel('Time (s)')
plt.ylabel('RR Interval (s)')
plt.grid(True)
plt.tight_layout()
plt.savefig("RR_Intervals_vs_Time.png")

plt.figure(figsize=(10,4))
plt.plot(hr_times, hr, 'mo-')
plt.title('Heart Rate (BPM) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Heart Rate (BPM)')
plt.grid(True)
plt.tight_layout()
plt.savefig("heartrate.png")
plt.show()

mean_hr = hr.mean()
sdnn    = rr_intervals.std()
diffs   = np.diff(rr_intervals)
rmssd   = np.sqrt(np.mean(diffs**2))
pnn50   = np.sum(np.abs(diffs) > 0.05) / len(diffs) * 100

#  Rate classification
if mean_hr < 60:
    rate_flag = "Bradycardia"
elif mean_hr > 100:
    rate_flag = "Tachycardia"
else:
    rate_flag = "Normal rate"

#  Variability check
var_flag = "Normal HRV"
if sdnn > 0.1 or rmssd > 0.06 or pnn50 > 20:
    var_flag = "Possible arrhythmia"

print(f"Mean HR: {mean_hr:.1f} BPM → {rate_flag}")
print(f"SDNN: {sdnn*1000:.1f} ms, RMSSD: {rmssd*1000:.1f} ms, pNN50: {pnn50:.1f}% → {var_flag}")

