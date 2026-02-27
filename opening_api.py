import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# 1. EXTRAER SEÑAL DE APERTURA -------------------------------------------------------

def extract_hand_opening_signal(video_path):

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(video_path)

    signal = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

            #wrist = np.array([landmarks[0].x, landmarks[0].y])

            #fingertips = np.array([
            #    [landmarks[8].x, landmarks[8].y],
            #    [landmarks[12].x, landmarks[12].y],
            #    [landmarks[16].x, landmarks[16].y],
            #    [landmarks[20].x, landmarks[20].y]
            #])

            wrist = np.array([landmarks[0].x,landmarks[0].y,landmarks[0].z])

            fingertips = np.array([
                [landmarks[8].x,  landmarks[8].y,  landmarks[8].z],
                [landmarks[12].x, landmarks[12].y, landmarks[12].z],
                [landmarks[16].x, landmarks[16].y, landmarks[16].z],
                [landmarks[20].x, landmarks[20].y, landmarks[20].z]
            ])

            #centroid = np.mean(fingertips, axis=0)
            #distance = np.linalg.norm(wrist - centroid)

            distances = [
                np.linalg.norm(wrist - fingertips[i])
                for i in range(4)
            ]

            distance = np.mean(distances)

            signal.append(distance)
            timestamps.append(ts)

    cap.release()
    hands.close()

    return np.array(signal), np.array(timestamps)

# 2. Creación del data frame -----------------------------------------------------------------
def build_opening_dataframe(signal, timestamps):

    df = pd.DataFrame({
        "time": timestamps * 1000,
        "amp": (signal / np.max(signal)) * 100 # (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 100
    })

    df = df[df['time'].diff().fillna(1) > 0]

    return df

def detect_cycles(df, min_distance_sec=0.3):

    t = df['time'].values / 1000
    signal = df['amp_smooth'].values

    dt = np.mean(np.diff(t))
    fs = 1 / dt

    peaks, _ = find_peaks(signal, prominence = 8,distance=int(fs * min_distance_sec))
    valleys, _ = find_peaks(-signal, prominence = 8 ,distance=int(fs * min_distance_sec))

    return peaks, valleys


# 3. PRE-PREOCESAMIENTO DE LA SEÑAL -----------------------------------------
def preprocess_signal(df, window_length=20, polyorder=3, interpolation_threshold=0.02):

    df = df.copy()

    t = df['time'].values / 1000.0 # Tiempo en segundos
    signal = df['amp'].values

    # Evaluar estabilidad temporal
    dt = np.diff(t)
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    cv = std_dt / mean_dt

    print("\n===== ESTABILIDAD TEMPORAL =====")
    print(f"Mean dt: {mean_dt:.6f}")
    print(f"STD dt: {std_dt:.6f}")
    print(f"CV: {cv:.4f}")

    # Interpolación si es necesario
    if cv > interpolation_threshold:
        print("⚠ Muestreo irregular detectado → aplicando interpolación")

        fs_uniform = 1 / mean_dt
        t_uniform = np.arange(t[0], t[-1], 1/fs_uniform)

        interpolator = interp1d(t, signal, kind='linear')
        signal = interpolator(t_uniform)
        t = t_uniform
    else:
        print("Muestreo suficientemente uniforme → no se interpola")

    # Suavizado Savitzky-Golay
    if len(signal) >= window_length:
        signal_smooth = savgol_filter(signal,window_length=window_length, polyorder=polyorder)
    else:
        signal_smooth = signal

    # Reconstruir DataFrame final
    df_processed = pd.DataFrame({'time': t * 1000,'amp_raw': signal,'amp_smooth': signal_smooth})

    return df_processed


# 4. Cálculo de métricas ------------------------------------------------------------
def compute_opening_metrics(df, peaks):

    t = df['time'].values / 1000
    signal = df['amp_smooth'].values

    # INTERVALOS Y FRECUENCIA
    intervals = np.diff(t[peaks])
    frequency = 1 / np.mean(intervals) if len(intervals) > 0 else 0
    sigma_interval = np.std(intervals) if len(intervals) > 0 else 0

    # AMPLITUD CICLO A CICLO
    amplitudes = []
    for i in range(len(peaks)-1):
        segment = signal[peaks[i]:peaks[i+1]]
        amplitudes.append(np.max(segment) - np.min(segment))

    amplitudes = np.array(amplitudes)

    mean_amplitude = np.mean(amplitudes) if len(amplitudes) > 0 else 0
    sigma_amplitude = np.std(amplitudes) if len(amplitudes) > 0 else 0

    # FATIGA TEMPRANA
    decrement_5 = amplitudes[0] - amplitudes[4] if len(amplitudes) >= 5 else 0
    decrement_7 = amplitudes[0] - amplitudes[6] if len(amplitudes) >= 7 else 0

    # FATIGA GLOBAL (Hilbert)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    global_fatigue = np.max(envelope) - np.min(envelope)

    return {
        "num_cycles": len(peaks),
        "frequency_hz": frequency,
        "sigma_interval": sigma_interval,
        "mean_amplitude": mean_amplitude,
        "sigma_amplitude": sigma_amplitude,
        "decrement_5": decrement_5,
        "decrement_7": decrement_7,
        "global_fatigue": global_fatigue
    }

# 5. FFT ------------------------------------------------------------------------------------------------
def compute_fft(df):

    t = df['time'].values / 1000.0
    signal = df['amp_smooth'].values

    dt = np.mean(np.diff(t))
    fs = 1 / dt

    signal_detrended = signal - np.mean(signal)

    N = len(signal_detrended)
    fft_vals = np.fft.fft(signal_detrended)
    fft_freq = np.fft.fftfreq(N, d=dt)

    positive = fft_freq > 0
    freqs = fft_freq[positive]
    spectrum = np.abs(fft_vals[positive])

    dominant_freq = freqs[np.argmax(spectrum)]

    print(f"\nFrecuencia de muestreo estimada: {fs:.2f} Hz")
    print(f"Frecuencia dominante del movimiento: {dominant_freq:.2f} Hz")

    return freqs, spectrum, dominant_freq

# 6. PROCESAMIENTO GENERAL ----------------------------------------
def process_hand_opening(video_path):

    signal, timestamps = extract_hand_opening_signal(video_path)

    if len(signal) < 10:
        return None

    df = build_opening_dataframe(signal, timestamps)
    df = preprocess_signal(df)

    peaks, valleys = detect_cycles(df)
    metrics = compute_opening_metrics(df, peaks)

    freqs, spectrum, dominant_freq = compute_fft(df)

    metrics["dominant_frequency_fft"] = dominant_freq
    metrics["peaks"] = peaks
    metrics["troughs"] = valleys
    metrics["df"] = df

    return metrics


# 7. ANÁLSIS DE MANOS -----------------------------------------------------------------
def analyze_opening(video_path):

    signal, timestamps = extract_hand_opening_signal(video_path)

    if len(signal) < 10:
        raise ValueError("Señal insuficiente")

    df = build_opening_dataframe(signal, timestamps)
    df_processed = preprocess_signal(df)

    peaks, valleys = detect_cycles(df_processed)
    metrics = compute_opening_metrics(df_processed, peaks)

    freqs, spectrum, dominant_freq = compute_fft(df_processed)
    
    # Agregar métricas FFT al dict
    metrics["dominant_frequency_fft"] = dominant_freq

    return {
        "df": df_processed,
        "metrics": metrics,
        "peaks": peaks,
        "troughs": valleys,
        "freqs": freqs,
        "spectrum": spectrum
    }


