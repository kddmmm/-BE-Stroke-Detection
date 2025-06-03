import parselmouth
import numpy as np

def safe_feature(value, fallback):
    try:
        val = float(value)
        if np.isnan(val) or val == 0.0:
            return fallback
        return val
    except (TypeError, ValueError):
        return fallback


def extract_features(filepath):
    try:
        snd = parselmouth.Sound(filepath)
        pitch = snd.to_pitch()
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = safe_feature(
            parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 0.01)
        jitter_ppq5 = safe_feature(
            parselmouth.praat.call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3), 0.005)
        shimmer_local = safe_feature(
            parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.6, 1.0), 0.02)
        shimmer_apq5 = safe_feature(
            parselmouth.praat.call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.6, 1.0), 0.02)

        f0_mean = safe_feature(
            parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz"), 120.0)
        f0_stdev = safe_feature(
            parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz"), 10.0)
        intensity = safe_feature(
            parselmouth.praat.call(snd.to_intensity(), "Get mean", 0, 0, "energy"), 60.0)

        hnr = safe_feature(
            parselmouth.praat.call(snd.to_harmonicity_cc(), "Get mean", 0, 0), 15.0)
        duration = safe_feature(snd.get_total_duration(), 1.5)
        f1 = safe_feature(
            parselmouth.praat.call(snd.to_formant_burg(), "Get value at time", 1, 0.5 * duration, 'Hertz', 'Linear'), 500.0)

        return [
            jitter_local, jitter_ppq5,
            shimmer_local, shimmer_apq5,
            f0_mean, f0_stdev,
            hnr, intensity,
            f1, duration
        ]
    except Exception as e:
        raise RuntimeError(f"[extract_features] {filepath} 실패: {e}")

