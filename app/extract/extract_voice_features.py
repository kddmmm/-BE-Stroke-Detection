import parselmouth
import numpy as np

def safe_feature(value, fallback):
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return fallback
        return val
    except (TypeError, ValueError):
        return fallback

def extract_features(filepath):
    try:
        snd = parselmouth.Sound(filepath)
        pitch = snd.to_pitch()
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)

        jitter_local = safe_feature(
            parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3), 0.01)
        jitter_ppq5 = safe_feature(
            parselmouth.praat.call(pp, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3), 0.005)
        f0_mean = safe_feature(
            parselmouth.praat.call(pitch, "Get mean", 0, 0, "Hertz"), 120.0)
        f0_std = safe_feature(
            parselmouth.praat.call(pitch, "Get standard deviation", 0, 0, "Hertz"), 10.0)
        hnr = safe_feature(
            parselmouth.praat.call(snd.to_harmonicity_cc(), "Get mean", 0, 0), 15.0)
        intensity = safe_feature(
            parselmouth.praat.call(snd.to_intensity(), "Get mean", 0, 0, "energy"), 60.0)
        duration = safe_feature(snd.get_total_duration(), 1.5)
        amplitude = safe_feature(np.abs(snd.values).max() if hasattr(snd, 'values') else None, 0.1)

        return [
            jitter_local, jitter_ppq5,
            f0_mean, f0_std,
            hnr, intensity,
            duration, amplitude
        ]
    except Exception as e:
        raise RuntimeError(f"[extract_features] {filepath} 실패: {e}")
