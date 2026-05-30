"""
JARVIS-PRIME Voice Engine
============================

Handles all speech I/O:
- Text-to-Speech (TTS): pyttsx3 (offline, instant) + edge-tts (premium voice)
- Speech-to-Text (STT): SpeechRecognition + OpenAI Whisper (local)
- Wake word detection: Continuous "JARVIS" listening
- Voice activity detection
- Robust hallucination filtering for Whisper

Designed for sub-second response on AMD Ryzen 7 7735HS.
"""
from __future__ import annotations

import asyncio
import math
import os
import queue
import re
import struct
import sys
import threading
import time
from typing import Any, Callable
import speech_recognition as sr


# ──────────────────────────────────────────────────────

def _compute_rms(audio_data: bytes, sample_width: int) -> float:
    """Compute RMS energy of raw audio bytes."""
    if sample_width == 2:
        fmt = "<{}h".format(len(audio_data) // 2)
        try:
            samples = struct.unpack(fmt, audio_data)
        except struct.error:
            return 0.0
    else:
        # fallback: treat as unsigned bytes
        samples = audio_data
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


class VoiceEngine:
    """
    Unified voice I/O engine for JARVIS-PRIME.
    Handles TTS, STT, and wake word detection.
    """

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower()
        self._tts_engine = None
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5  # Wait 0.5s of silence before cutting off for faster response

        # Disable dynamic adjustment — it drifts too low in quiet rooms
        self.recognizer.dynamic_energy_threshold = False
        self.microphone = sr.Microphone()

        # Calibrate once at startup
        print("[VOICE] Calibrating microphone for ambient noise (2 seconds)...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2.0)

        # Enforce a strict floor so Whisper never processes pure silence.
        # 300 works well for typical laptop/desktop mics; raise to 400-500
        # if you still get phantom phrases.
        self.recognizer.energy_threshold = max(
            self.recognizer.energy_threshold, 300
        )
        print(f"[VOICE] Energy threshold set to: {self.recognizer.energy_threshold:.0f}")

        self._is_listening = False
        self._speech_queue: queue.Queue[str] = queue.Queue()
        self._tts_lock = threading.Lock()
        self._callbacks: dict[str, Callable] = {}

        # Timestamp of the last TTS utterance — used to avoid hearing our own voice
        self._last_speak_end: float = 0.0

        # TTS settings
        self.tts_rate = 185       # Words per minute
        self.tts_volume = 0.9
        self.tts_voice_id = None  # Auto-select best voice

        self._tts_queue = queue.Queue()
        self._tts_ready = threading.Event()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._tts_thread.start()

        self._init_stt()

    # ──────────────────────────────────────────────────
    # Text-to-Speech
    # ──────────────────────────────────────────────────

    def _tts_worker(self) -> None:
        """Background thread to process TTS requests using raw SAPI."""
        try:
            import pythoncom
            import win32com.client

            # Initialize COM for this background thread
            pythoncom.CoInitialize()
            engine = win32com.client.Dispatch("SAPI.SpVoice")

            # Adjust rate (SAPI rate is -10 to 10, default is 0)
            engine.Rate = 2
            engine.Volume = int(self.tts_volume * 100)

            self._tts_engine = engine
            print("[VOICE] TTS engine: SAPI.SpVoice (offline, instant)")
            self._tts_ready.set()
        except Exception as e:
            print(f"[VOICE] TTS init failed: {e}")
            self._tts_engine = None
            self._tts_ready.set()
            return

        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                engine.Speak(text)
            except Exception as e:
                print(f"[VOICE] TTS Error: {e}")
            finally:
                # Record when we stopped speaking so the mic can ignore echo
                self._last_speak_end = time.monotonic()
                self._tts_queue.task_done()

        try:
            pythoncom.CoUninitialize()
        except Exception:
            pass

    def speak(self, text: str, block: bool = True) -> None:
        """
        Speak text aloud by passing it to the TTS worker thread.
        """
        if not text:
            return

        self._tts_ready.wait()

        if self._tts_engine is None:
            print(f"[JARVIS]: {text}")
            return

        self._tts_queue.put(text)
        if block:
            self._tts_queue.join()

    def speak_async(self, text: str) -> None:
        """Speak without blocking."""
        self.speak(text, block=False)

    # ──────────────────────────────────────────────────
    # Speech-to-Text
    # ──────────────────────────────────────────────────

    def _init_stt(self) -> None:
        """Initialize speech recognition."""
        try:
            print("[VOICE] STT engine: Google Speech Recognition (Fast, accurate, no hallucinations)")
        except Exception as e:
            print(f"[VOICE] STT init failed: {e}")
            print("[VOICE] Tip: Install PyAudio: pip install pyaudio")

    def listen(self, timeout: float = 5.0, phrase_limit: float = 15.0) -> str | None:
        """
        Listen for speech and return transcribed text.
        Returns None if nothing detected or recognition fails.
        """
        if self.recognizer is None or self.microphone is None:
            return None

        try:
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )

            # ── Pre-filter: check if audio has real energy ──
            # If the raw audio is mostly silence, skip STT entirely.
            try:
                rms = _compute_rms(audio.get_raw_data(), audio.sample_width)
                if rms < 200:
                    return None
            except Exception:
                pass  # If we can't compute RMS, let STT decide

            # ── Google recognition (Fast, online, no hallucinations) ──
            try:
                text = self.recognizer.recognize_google(audio)
                return text.strip() if text else None
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                print("[VOICE] Google STT unreachable, falling back to Sphinx (offline)...")
                # Fallback: try offline pocketsphinx if available
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text.strip() if text else None
                except Exception:
                    return None

        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None

    # ──────────────────────────────────────────────────
    # Wake Word Detection
    # ──────────────────────────────────────────────────

    def start_wake_word_listener(
        self,
        on_wake: Callable[[], None] | None = None,
        on_command: Callable[[str], None] | None = None,
    ) -> None:
        """
        Start continuous wake word listening in a background thread.

        Flow:
        1. Listen for "JARVIS" wake word
        2. Play acknowledgment sound
        3. Listen for the actual command
        4. Call on_command callback with the transcribed text
        """
        self._is_listening = True
        self._callbacks['on_wake'] = on_wake
        self._callbacks['on_command'] = on_command

        thread = threading.Thread(target=self._wake_loop, daemon=True)
        thread.start()
        print(f'[VOICE] Wake word listener active. Say "{self.wake_word.upper()}" to activate.')

    def _wake_loop(self) -> None:
        """Main wake word detection loop."""
        _WAKE_VARIANTS = [self.wake_word, "travis", "harvis"]

        while self._is_listening:
            try:
                # ── Cooldown: skip listening if JARVIS just finished speaking ──
                # This prevents the mic from picking up JARVIS's own TTS output
                # and mistaking it for user speech.
                since_speak = time.monotonic() - self._last_speak_end
                if since_speak < 0.8:
                    time.sleep(0.8 - since_speak)
                    continue

                text = self.listen(timeout=3.0, phrase_limit=4.0)

                if not text:
                    continue

                print(f"  [DEBUG-STT] Heard: '{text}'")

                lower_text = text.lower()

                # Check if any wake-word variant appears
                matched_wake = None
                for w in _WAKE_VARIANTS:
                    if w in lower_text:
                        matched_wake = w
                        break

                if not matched_wake:
                    continue

                # Wake word detected!
                print(f"  [WAKE WORD] Detected in: '{text}'")

                idx = lower_text.find(matched_wake)
                remainder = text[idx + len(matched_wake):].strip()
                # Strip leading punctuation from remainder (e.g., "jarvis, open chrome")
                remainder = remainder.lstrip(",.!? ")

                command = None
                if len(remainder) > 3:
                    # User spoke the command in the same breath
                    command = remainder
                    if self._callbacks.get('on_wake'):
                        self._callbacks['on_wake']()
                else:
                    # User just said "Jarvis", so we respond and wait
                    if self._callbacks.get('on_wake'):
                        self._callbacks['on_wake']()
                    else:
                        self.speak("Yes, I'm here.", block=True)

                    # Now listen for the actual command (longer timeout)
                    command = self.listen(timeout=8.0, phrase_limit=30.0)

                if command and self._callbacks.get('on_command'):
                    self._callbacks['on_command'](command)

            except Exception as e:
                import traceback
                print(f"[VOICE] Error in wake loop: {e}")
                traceback.print_exc()
                time.sleep(0.3)

    def stop_listening(self) -> None:
        """Stop the wake word listener."""
        self._is_listening = False

    # ──────────────────────────────────────────────────
    # Status
    # ──────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Get voice engine status."""
        return {
            "tts_available": self._tts_engine is not None,
            "tts_engine": "SAPI.SpVoice" if self._tts_engine else "none",
            "tts_rate": self.tts_rate,
            "stt_available": self.recognizer is not None,
            "stt_engine": "Whisper (base.en)" if self.recognizer else "none",
            "microphone_available": self.microphone is not None,
            "wake_word": self.wake_word,
            "is_listening": self._is_listening,
            "energy_threshold": self.recognizer.energy_threshold if self.recognizer else 0,
        }
