import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.signal import find_peaks

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# 1️. INFORMACIÓN DEL VIDEO -------------------------------------------------
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Error abriendo el video")

    fps_metadata = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps_metadata if fps_metadata > 0 else 0

    print("\n===== INFORMACIÓN DEL VIDEO =====")
    print(f"FPS (metadata): {fps_metadata}")
    print(f"Total frames: {total_frames}")
    print(f"Duración estimada (s): {duration:.3f}")

    return cap, fps_metadata


# 2️. EXTRACCIÓN DE LANDMARKS -------------------------------------------------------
def extract_landmarks(cap):
    mpHands = mp.solutions.hands
    allList = []
    timestamps_fps = []

    with mpHands.Hands(
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, img = cap.read()
            img = cv2.resize(img, (1280, 720))
            if not success:
                break

            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps_fps.append(ts / 1000)

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            lmList = []

            if results.multi_hand_landmarks:
                handLms = results.multi_hand_landmarks[0]

                for id, lm in enumerate(handLms.landmark):
                    if id == 4 or id == 8:
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy, ts])

            if len(lmList) == 2:
                allList.append(lmList)

    cap.release()
    return np.array(allList), np.array(timestamps_fps)


# 3. ANÁLISIS FPS REAL -------------------------------------------------
def analyze_real_fps(timestamps):
    dt = np.diff(timestamps)

    if len(dt) == 0:
        print("No se pudo calcular FPS real.")
        return None

    fps_real = 1 / dt
    
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    cv = std_dt / mean_dt

    print("\n===== ANÁLISIS DE FPS REAL =====")
    print(f"FPS promedio real: {np.mean(fps_real):.2f}")
    print(f"Desviación estándar FPS: {np.std(fps_real):.2f}")
    print(f"FPS mínimo: {np.min(fps_real):.2f}")
    print(f"FPS máximo: {np.max(fps_real):.2f}")
    print(f"CV del Δt: {cv:.4f}")

    return fps_real


# 4️. CONSTRUIR DATAFRAME DE DISTANCIAS ---------------------------------------------------------
def build_dataframe(np_allList):
    length = []

    for i in range(np_allList.shape[0]):
        x1, y1 = np_allList[i][0][1], np_allList[i][0][2]
        x2, y2 = np_allList[i][1][1], np_allList[i][1][2]
        ts = np_allList[i][0][3]

        dist = math.hypot(x2 - x1, y2 - y1)
        length.append([ts, dist])

    df = pd.DataFrame(length, columns=['time', 'dist'])

    # eliminar timestamps duplicados
    df['diff'] = df['time'].diff()
    df = df[df['diff'] > 0]
    df.drop(columns=['diff'], inplace=True)

    if df.empty:
        raise ValueError("DataFrame vacío después de limpiar timestamps.")

    # Normalización
    df['amp'] = (df['dist'] / df['dist'].max()) * 100
    #df.drop(columns=['dist'], inplace=True)

    # Limitar a primeros 10 segundos
    df = df[df['time'] <= 10000]

    return df


# 5. PRE-PREOCESAMIENTO DE LA SEÑAL -----------------------------------------
def preprocess_signal(df, window_length=9, polyorder=3, interpolation_threshold=0.02):

    df = df.copy()

    t = df['time'].values / 1000.0 # Tiempo en segundos
    signal_amp = df['amp'].values
    signal_dist = df['dist'].values

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

        interp_dist = interp1d(t, signal_dist, kind='linear')
        interp_amp = interp1d(t, signal_amp, kind='linear')

        signal_dist = interp_dist(t_uniform)
        signal_amp = interp_amp(t_uniform)
        t = t_uniform
    else:
        print("Muestreo suficientemente uniforme → no se interpola")

    # Suavizado Savitzky-Golay
    if len(signal_dist) >= window_length:
        dist_smooth = savgol_filter(signal_dist, window_length, polyorder)
        amp_smooth = savgol_filter(signal_amp, window_length, polyorder)
    else:
        dist_smooth = signal_dist
        amp_smooth = signal_amp

    # Reconstruir DataFrame final
    #df_processed = pd.DataFrame({'time': t * 1000,'amp_raw': signal,'amp_smooth': signal_smooth})

    df_processed = pd.DataFrame({
        'time': t * 1000,
        'dist_smooth': dist_smooth,
        'amp_smooth': amp_smooth
    })

    return df_processed


# 6. CÁLCULO DE MÉTRICAS --------------------------------------------
def compute_temporal_metrics(df, prominence=30):

    signal = df['amp_smooth'].values
    signal_amp = df['amp_smooth'].values
    signal_dist = df['dist_smooth'].values
    time_sec = df['time'].values / 1000.0

    # Detectar máximos (picos)
    peaks, _ = find_peaks(signal, prominence=prominence)
    # Detectar mínimos (valles / troughs)
    troughs, _ = find_peaks(-signal, prominence=prominence)

    # Amplitudes en los picos
    amps = df.iloc[peaks]['amp_smooth'].reset_index(drop=True)
    tap_count = len(amps)

    mean_amp = amps.mean()
    std_amp = amps.std()

    # Padding hasta 10 taps
    #while len(amps) < 10:
    #    amps.loc[len(amps)] = 0

    # Diferencias de amplitud
    #diff3  = amps.iloc[0] - amps.iloc[2]
    #diff5  = amps.iloc[0] - amps.iloc[4]
    #diff7  = amps.iloc[0] - amps.iloc[6]
    #diff10 = amps.iloc[0] - amps.iloc[9]

    amps_padded = np.copy(amps)
    while len(amps_padded) < 10:
        amps_padded = np.append(amps_padded, 0)

    diff3  = amps_padded[0] - amps_padded[2]
    diff5  = amps_padded[0] - amps_padded[4]
    diff7  = amps_padded[0] - amps_padded[6]
    diff10 = amps_padded[0] - amps_padded[9]

    # Frecuencia método 1 (Npicos / T)
    total_time = time_sec[-1] - time_sec[0] if len(time_sec) > 1 else 0
    ft_simple = tap_count / total_time if total_time > 0 else 0

    # Frecuencia método 2 (1 / mean ITI)
    if tap_count > 1:
        iti = np.diff(time_sec[peaks])
        mean_iti = np.mean(iti)
        std_iti = np.std(iti)
        ft_iti = 1 / mean_iti if mean_iti > 0 else 0
    else:
        mean_iti = 0
        std_iti = 0
        ft_iti = 0


    # Velocidad media (derivada discreta)
    if len(signal_dist) > 1:
        velocity = np.diff(signal_dist) / np.diff(time_sec)
        mean_velocity = np.mean(np.abs(velocity))
        max_velocity = np.max(np.abs(velocity))

        # Normalización por amplitud media
        mean_amplitude_dist = np.mean(signal_dist)
        mean_velocity_rel = mean_velocity / mean_amplitude_dist
    else:
        mean_velocity = 0
        max_velocity = 0

    # Pendiente de amplitud (fatiga)
    if tap_count > 1:
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(range(tap_count), amps)
    else:
        slope = 0

    results = {
        "tap_count": tap_count,
        "mean_amp": mean_amp,
        "std_amp": std_amp,
        "diff3": diff3,
        "diff5": diff5,
        "diff7": diff7,
        "diff10": diff10,
        "ft_simple": ft_simple,
        "ft_iti": ft_iti,
        "mean_iti": mean_iti,
        "std_iti": std_iti,
        "mean_velocity": mean_velocity_rel,
        "slope_amplitude": slope,
        "peaks": peaks,
        "troughs": troughs
    }

    return results


# 7. ANÁLISIS FOURIER -------------------------------------------------
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

    # Energía total
    total_energy = np.sum(spectrum)

    # Energía alrededor de f0 (±0.5 Hz)
    band_width = 0.5
    band_mask = (freqs >= dominant_freq - band_width) & \
                (freqs <= dominant_freq + band_width)

    energy_f0 = np.sum(spectrum[band_mask])
    regularity_index = energy_f0 / total_energy if total_energy > 0 else 0

    return {
        "freqs": freqs,
        "spectrum": spectrum,
        "dominant_freq": dominant_freq,
        "total_energy": total_energy,
        "regularity_index": regularity_index
    }


# 8. GRÁFICAS -------------------------------------------------

def plot_signal(df, peaks, troughs, hand_label):

    plt.figure(figsize=(10,5))
    plt.plot(df['time'], df['amp_smooth'])
    plt.plot(df['time'].iloc[peaks], df['amp_smooth'].iloc[peaks], "o")
    plt.plot(df['time'].iloc[troughs], df['amp_smooth'].iloc[troughs],"x")
    plt.xlabel("Tiempo (msec)")
    plt.ylabel("Amplitud (%)")
    plt.title(f"Finger Tapping - {hand_label}")
    plt.grid(True)
    plt.show()


def plot_spectrum(freqs, spectrum, hand_label):

    plt.figure(figsize=(10,5))
    plt.plot(freqs, spectrum)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.title(f"Espectro de Frecuencia - {hand_label}")
    plt.grid(True)
    plt.show()


# PROCESAR UN VIDEO ---------------------------------------------------------

def analyze_fingertap(video_path):
    
    cap, _ = get_video_info(video_path)
    landmarks, timestamps = extract_landmarks(cap)

    if len(landmarks) == 0:
        raise ValueError("No se detectaron manos")

    df = build_dataframe(landmarks)
    df_processed = preprocess_signal(df)

    temporal_results = compute_temporal_metrics(df_processed)
    fft_results = compute_fft(df_processed)

    # Extraer peaks y troughs fuera del dict
    peaks = temporal_results.pop("peaks")
    troughs = temporal_results.pop("troughs")

    if fft_results is not None:
        temporal_results.update({
            "dominant_freq": fft_results["dominant_freq"],
            "total_energy": fft_results["total_energy"],
            "regularity_index": fft_results["regularity_index"]
        })

    return {
        "time": df_processed["time"].tolist(),
        "distance": df_processed["distance"].tolist(),
        "metrics": temporal_results,
        "peaks": peaks.tolist(),
        "troughs": troughs.tolist(),
        "freqs": fft_results["freqs"].tolist() if fft_results else [],
        "spectrum": fft_results["spectrum"].tolist() if fft_results else []
    }


# GUARDADO DE SEÑALES ------------------------------
def build_metrics_dataframe(metrics_dict, patient_id, estado, movement, hand, attempt):

    row = {
        "patient_id": patient_id,
        "estado": estado,
        "movement": movement,
        "hand": hand,
        "attempt": attempt,
        "tap_count": metrics_dict["tap_count"],
        "mean_amp": metrics_dict["mean_amp"],
        "std_amp": metrics_dict["std_amp"],
        "diff3": metrics_dict["diff3"],
        "diff5": metrics_dict["diff5"],
        "diff7": metrics_dict["diff7"],
        "diff10": metrics_dict["diff10"],
        "ft_simple": metrics_dict.get("ft_simple", 0),
        "ft_iti": metrics_dict.get("ft_iti", 0),
        "mean_iti": metrics_dict.get("mean_iti", 0),
        "std_iti": metrics_dict.get("std_iti", 0),
        "mean_velocity": metrics_dict.get("mean_velocity", 0),
        "slope_amplitude": metrics_dict.get("slope_amplitude", 0),
        "dominant_freq": metrics_dict.get("dominant_freq", 0),
        "total_energy": metrics_dict.get("total_energy", 0),
        "regularity_index": metrics_dict.get("regularity_index", 0)
    }

    return pd.DataFrame([row])

def build_signal_dataframe(df_processed, patient_id, estado, movement, hand, attempt):

    df = df_processed.copy()

    df["patient_id"] = patient_id
    df["estado"] = estado
    df["movement"] = movement
    df["hand"] = hand
    df["attempt"] = attempt

    return df[[
        "patient_id",
        "estado",
        "movement",
        "hand",
        "attempt",
        "time",
        "amp_smooth"
    ]]

import os

def save_results(patient_id, signal_df, metrics_df):

    patient_folder = os.path.join("data", patient_id)
    os.makedirs(patient_folder, exist_ok=True)

    # Guardar señales del paciente
    signal_path = os.path.join(patient_folder, "signals_all_attempts.csv")

    if os.path.exists(signal_path):
        signal_df.to_csv(signal_path, mode='a', header=False, index=False)
    else:
        signal_df.to_csv(signal_path, index=False)

    # Guardar métricas del paciente
    metrics_path = os.path.join(patient_folder, "metrics_all_attempts.csv")

    if os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_path, index=False)

    # Guardar en CSV global
    global_path = os.path.join("data", "ALL_PATIENTS_METRICS.csv")

    if os.path.exists(global_path):
        metrics_df.to_csv(global_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(global_path, index=False)

