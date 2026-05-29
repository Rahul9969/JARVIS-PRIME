"""
JARVIS-PRIME Audio Perception Module
=======================================

Provides:
- Audio feature extraction (MFCC, spectrogram, zero-crossing)
- Speech command recognition framework
- Wake word detection pipeline
- Audio signal analysis

Phase 3: Pure NumPy DSP (no model downloads)
Phase 4+: Whisper integration for full STT
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


class AudioAnalyzer:
    """
    Audio signal processing and feature extraction.
    Pure NumPy implementation — no librosa/torch required.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def generate_test_signal(
        self,
        frequencies: list[float] | None = None,
        duration_s: float = 1.0,
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """Generate a test audio signal (sum of sinusoids + noise)."""
        if frequencies is None:
            frequencies = [440.0, 880.0]  # A4 + A5

        t = np.linspace(0, duration_s, int(self.sr * duration_s))
        signal = np.zeros_like(t)

        for freq in frequencies:
            signal += np.sin(2 * np.pi * freq * t)

        signal /= len(frequencies)
        signal += noise_level * np.random.randn(len(t))
        return signal

    def compute_spectrogram(
        self,
        signal: np.ndarray,
        window_size: int = 512,
        hop_size: int = 256,
    ) -> dict[str, Any]:
        """Compute short-time Fourier transform (STFT) spectrogram."""
        n_frames = (len(signal) - window_size) // hop_size + 1
        if n_frames <= 0:
            return {"error": "Signal too short for window size"}

        # Hann window
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / window_size))

        spectrogram = np.zeros((window_size // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * hop_size
            frame = signal[start:start + window_size] * window
            fft = np.fft.rfft(frame)
            spectrogram[:, i] = np.abs(fft)

        # Power spectrogram in dB
        power_db = 20 * np.log10(spectrogram + 1e-10)

        return {
            "n_frames": n_frames,
            "n_frequency_bins": window_size // 2 + 1,
            "frequency_resolution_Hz": self.sr / window_size,
            "time_resolution_s": hop_size / self.sr,
            "max_frequency_Hz": self.sr / 2,
            "peak_frequency_Hz": round(float(np.argmax(spectrogram.mean(axis=1)) * self.sr / window_size), 1),
            "dynamic_range_dB": round(float(power_db.max() - power_db.min()), 1),
        }

    def compute_mfcc(
        self,
        signal: np.ndarray,
        n_mfcc: int = 13,
        n_fft: int = 512,
        n_mels: int = 40,
    ) -> dict[str, Any]:
        """
        Compute Mel-frequency cepstral coefficients (MFCC).
        Simplified implementation.
        """
        # Compute power spectrum
        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n_fft) / n_fft))
        frames = []

        for i in range(0, len(signal) - n_fft, n_fft // 2):
            frame = signal[i:i + n_fft] * window
            power = np.abs(np.fft.rfft(frame)) ** 2
            frames.append(power)

        if not frames:
            return {"error": "Signal too short"}

        power_spectrum = np.array(frames)

        # Mel filterbank (simplified triangular)
        mel_low = 0
        mel_high = 2595 * math.log10(1 + self.sr / 2 / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        # Apply filterbank
        n_bins = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_bins))
        for i in range(n_mels):
            left, center, right = bin_points[i], bin_points[i+1], bin_points[i+2]
            for j in range(left, min(center, n_bins)):
                filterbank[i, j] = (j - left) / max(center - left, 1)
            for j in range(center, min(right, n_bins)):
                filterbank[i, j] = (right - j) / max(right - center, 1)

        mel_spectrum = np.dot(power_spectrum, filterbank.T)
        mel_spectrum = np.log(mel_spectrum + 1e-10)

        # DCT to get MFCCs
        mfcc = np.zeros((mel_spectrum.shape[0], n_mfcc))
        for i in range(n_mfcc):
            mfcc[:, i] = np.sum(
                mel_spectrum * np.cos(np.pi * i * (2 * np.arange(n_mels) + 1) / (2 * n_mels)),
                axis=1,
            )

        return {
            "n_mfcc": n_mfcc,
            "n_frames": len(frames),
            "mean_coefficients": [round(float(x), 4) for x in mfcc.mean(axis=0)],
            "std_coefficients": [round(float(x), 4) for x in mfcc.std(axis=0)],
        }

    def analyze_signal(self, signal: np.ndarray) -> dict[str, Any]:
        """Comprehensive signal analysis."""
        duration = len(signal) / self.sr

        # Zero-crossing rate
        zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

        # RMS energy
        rms = np.sqrt(np.mean(signal**2))

        # Peak frequency via FFT
        fft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), 1 / self.sr)
        peak_freq = freqs[np.argmax(fft)]

        # Spectral centroid
        spectral_centroid = np.sum(freqs * fft) / np.sum(fft)

        return {
            "duration_s": round(duration, 3),
            "sample_rate": self.sr,
            "n_samples": len(signal),
            "rms_amplitude": round(float(rms), 6),
            "peak_amplitude": round(float(np.max(np.abs(signal))), 6),
            "zero_crossing_rate": round(float(zcr), 4),
            "peak_frequency_Hz": round(float(peak_freq), 1),
            "spectral_centroid_Hz": round(float(spectral_centroid), 1),
            "is_speech_likely": 80 < peak_freq < 3000 and zcr > 0.02,
        }


class WakeWordDetector:
    """
    Simple wake word detection framework.
    Uses template matching with MFCC features.

    Phase 3: Template-based (low accuracy, zero downloads)
    Phase 4: Whisper-based or custom neural net
    """

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word
        self.analyzer = AudioAnalyzer()
        self.is_listening = False

    def status(self) -> dict[str, Any]:
        return {
            "wake_word": self.wake_word,
            "is_listening": self.is_listening,
            "backend": "template_matching",
            "note": "Set JARVIS_WHISPER=1 to enable Whisper STT (requires model download)",
            "upgrade_path": [
                "Phase 3: Template matching (current)",
                "Phase 4: Whisper.cpp local STT",
                "Phase 5: Custom neural wake word + streaming STT",
            ],
        }

    def start(self) -> dict[str, Any]:
        self.is_listening = True
        return {"status": "listening", "wake_word": self.wake_word}

    def stop(self) -> dict[str, Any]:
        self.is_listening = False
        return {"status": "stopped"}
