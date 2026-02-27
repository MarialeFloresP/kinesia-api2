import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import find_peaks, hilbert
import math
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ==========================
# 1️⃣ EXTRAER SEÑAL ANGULAR


def extract_pronation_signal(video_path):

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    angle_signal = []
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark

            thumb = np.array([landmarks[4].x, landmarks[4].y])
            pinky = np.array([landmarks[20].x, landmarks[20].y])

            vector = pinky - thumb

            # Ángulo respecto al eje horizontal
            angle = math.degrees(math.atan2(vector[1], vector[0]))

            angle_signal.append(angle)

            timestamps.append(ts)

    cap.release()
    hands.close()

    # AQUÍ HACEMOS EL UNWRAP (fuera del loop)
    angle_signal = np.array(angle_signal)

    angle_signal = np.unwrap(np.radians(angle_signal))
    angle_signal = np.degrees(angle_signal)

    return np.array(angle_signal), np.array(timestamps), fps



def build_pronation_dataframe(signal, timestamps):

    df = pd.DataFrame({
        'time': timestamps * 1000,  # en ms
        'amp': signal
    })

    df = df.sort_values("time")
    df = df[df['time'].diff().fillna(1) > 0]
    df = df.reset_index(drop=True)

    return df



# ==========================
# 2️⃣ DETECTAR CICLOS ANGULARES
# ==========================

def detect_angular_cycles(signal, fps):

    # Suavizado básico
    signal = (signal - np.mean(signal)) / np.std(signal)

    peaks, _ = find_peaks(signal, distance=fps*0.3,prominence=20)
    valleys, _ = find_peaks(-signal, distance=fps*0.3, prominence=20)

    return peaks, valleys


# ==========================
# 3️⃣ CALCULAR MÉTRICAS
# ==========================

def compute_angular_metrics(signal, peaks, fps):

    if len(peaks) < 2:
        return None

    # Intervalos temporales
    intervals = np.diff(peaks) / fps
    frequency = 1 / np.mean(intervals)
    sigma_I = np.std(intervals)

    # Amplitud angular por ciclo
    amplitudes = []

    for i in range(len(peaks)-1):
        segment = signal[peaks[i]:peaks[i+1]]
        amplitudes.append(np.max(segment) - np.min(segment))

    amplitudes = np.array(amplitudes)

    mean_amplitude = np.mean(amplitudes)
    sigma_A = np.std(amplitudes)

    # Fatiga temprana
    decrement_5 = amplitudes[0] - amplitudes[4] if len(amplitudes) >= 5 else 0
    decrement_7 = amplitudes[0] - amplitudes[6] if len(amplitudes) >= 7 else 0

    # Fatiga global
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)
    global_fatigue = np.max(envelope) - np.min(envelope)


    return {
        "num_cycles": len(peaks),
        "frequency_hz": frequency,
        "sigma_interval": sigma_I,
        "mean_angular_amplitude_deg": mean_amplitude,
        "sigma_angular_amplitude": sigma_A,
        "decrement_5": decrement_5,
        "decrement_7": decrement_7,
        "global_fatigue": global_fatigue
    }




# 3. PRE-PREOCESAMIENTO DE LA SEÑAL -----------------------------------------
def preprocess_signal(df, window_length=17, polyorder=3, interpolation_threshold=0.02):

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

# FFT ------------------------------------------------------
def compute_fft(df):

    t = df['time'].values / 1000.0
    signal = df['amp_smooth'].values

    if len(signal) < 5:
        return None

    dt = np.mean(np.diff(t))
    fs = 1 / dt

    # Quitar componente DC
    signal_detrended = signal - np.mean(signal)

    N = len(signal_detrended)
    fft_vals = np.fft.fft(signal_detrended)
    fft_freq = np.fft.fftfreq(N, d=dt)

    positive = fft_freq > 0
    freqs = fft_freq[positive]
    spectrum = np.abs(fft_vals[positive])

    if len(spectrum) == 0:
        return None

    dominant_freq = freqs[np.argmax(spectrum)]

    total_energy = np.sum(spectrum)

    # Índice de regularidad ±0.5 Hz
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



def analyze_pronation_supination(video_path):

    signal, timestamps, fps = extract_pronation_signal(video_path)

    if len(signal) < 10:
        return None

    # 1️⃣ Construir DataFrame
    df = build_pronation_dataframe(signal, timestamps)

    # 2️⃣ Preprocesamiento (TU función)
    df_processed = preprocess_signal(df)

    signal_filtered = df_processed['amp_smooth'].values

    # 3️⃣ Detectar ciclos
    peaks, valleys = detect_angular_cycles(signal_filtered, fps)

    # 4️⃣ Calcular métricas
    metrics = compute_angular_metrics(signal_filtered, peaks, fps)

    return metrics


# -----------------------------------------------------------------

def analyze_pronation(video_path):

    signal, timestamps, fps = extract_pronation_signal(video_path)

    if len(signal) < 10:
        raise ValueError("Señal insuficiente")

    df = build_pronation_dataframe(signal, timestamps)
    df_processed = preprocess_signal(df)

    signal_filtered = df_processed['amp_smooth'].values

    peaks, valleys = detect_angular_cycles(signal_filtered, fps)

    metrics = compute_angular_metrics(signal_filtered, peaks, fps)

    # FFT
    fft_results = compute_fft(df_processed)

    if fft_results is not None:
        metrics.update({
            "dominant_frequency_fft": fft_results["dominant_freq"],
            "total_energy": fft_results["total_energy"],
            "regularity_index": fft_results["regularity_index"]
        })

        freqs = fft_results["freqs"]
        spectrum = fft_results["spectrum"]
    else:
        freqs = None
        spectrum = None

    return {
        "df": df_processed,
        "metrics": metrics,
        "peaks": peaks,
        "troughs": valleys,
        "freqs": freqs,
        "spectrum": spectrum
    }