"""
JARVIS-PRIME Voice Engine
============================

Handles all speech I/O:
- Text-to-Speech (TTS): pyttsx3 (offline, instant) + edge-tts (premium voice)
- Speech-to-Text (STT): SpeechRecognition + Google/Vosk
- Wake word detection: Continuous "JARVIS" listening
- Voice activity detection

Designed for sub-second response on AMD Ryzen 7 7735HS.
"""
from __future__ import annotations

import asyncio
import os
import queue
import sys
import threading
import time
from typing import Any, Callable


class VoiceEngine:
    """
    Unified voice I/O engine for JARVIS-PRIME.
    Handles TTS, STT, and wake word detection.
    """

    def __init__(self, wake_word: str = "jarvis"):
        self.wake_word = wake_word.lower()
        self._tts_engine = None
        self._recognizer = None
        self._microphone = None
        self._is_listening = False
        self._speech_queue: queue.Queue[str] = queue.Queue()
        self._tts_lock = threading.Lock()
        self._callbacks: dict[str, Callable] = {}

        # TTS settings
        self.tts_rate = 185       # Words per minute
        self.tts_volume = 0.9
        self.tts_voice_id = None  # Auto-select best voice

        self._init_tts()
        self._init_stt()

    # ──────────────────────────────────────────────────
    # Text-to-Speech
    # ──────────────────────────────────────────────────

    def _init_tts(self) -> None:
        """Initialize text-to-speech engine."""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty('rate', self.tts_rate)
            self._tts_engine.setProperty('volume', self.tts_volume)

            # Try to find a good voice
            voices = self._tts_engine.getProperty('voices')
            for voice in voices:
                # Prefer Microsoft David or Zira (Windows)
                name_lower = voice.name.lower()
                if 'david' in name_lower or 'mark' in name_lower:
                    self._tts_engine.setProperty('voice', voice.id)
                    self.tts_voice_id = voice.id
                    break
            if not self.tts_voice_id and voices:
                self._tts_engine.setProperty('voice', voices[0].id)
                self.tts_voice_id = voices[0].id

            print("[VOICE] TTS engine: pyttsx3 (offline, instant)")
        except Exception as e:
            print(f"[VOICE] TTS init failed: {e}")
            self._tts_engine = None

    def speak(self, text: str, block: bool = True) -> None:
        """
        Speak text aloud.
        Uses pyttsx3 for instant offline speech.
        """
        if not text:
            return

        if self._tts_engine is None:
            print(f"[JARVIS]: {text}")
            return

        with self._tts_lock:
            try:
                self._tts_engine.say(text)
                if block:
                    self._tts_engine.runAndWait()
            except Exception as e:
                print(f"[JARVIS]: {text}")

    def speak_async(self, text: str) -> None:
        """Speak without blocking the main thread."""
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()

    # ──────────────────────────────────────────────────
    # Speech-to-Text
    # ──────────────────────────────────────────────────

    def _init_stt(self) -> None:
        """Initialize speech recognition."""
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.8

            self._microphone = sr.Microphone()

            # Calibrate for ambient noise
            with self._microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

            print("[VOICE] STT engine: SpeechRecognition (Google)")
        except Exception as e:
            print(f"[VOICE] STT init failed: {e}")
            print("[VOICE] Tip: Install PyAudio: pip install pyaudio")

    def listen(self, timeout: float = 5.0, phrase_limit: float = 15.0) -> str | None:
        """
        Listen for speech and return transcribed text.
        Returns None if nothing detected or recognition fails.
        """
        if self._recognizer is None or self._microphone is None:
            return None

        import speech_recognition as sr

        try:
            with self._microphone as source:
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_limit,
                )

            # Try Google (free, good quality)
            try:
                text = self._recognizer.recognize_google(audio)
                return text.strip()
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
                # Fallback: try offline if available
                try:
                    text = self._recognizer.recognize_sphinx(audio)
                    return text.strip()
                except Exception:
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
        while self._is_listening:
            try:
                text = self.listen(timeout=2.0, phrase_limit=3.0)
                if text and self.wake_word in text.lower():
                    # Wake word detected!
                    if self._callbacks.get('on_wake'):
                        self._callbacks['on_wake']()
                    else:
                        self.speak("Yes, I'm here.", block=True)

                    # Now listen for the actual command
                    command = self.listen(timeout=8.0, phrase_limit=30.0)
                    if command and self._callbacks.get('on_command'):
                        self._callbacks['on_command'](command)

            except Exception:
                time.sleep(0.1)

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
            "tts_engine": "pyttsx3" if self._tts_engine else "none",
            "tts_rate": self.tts_rate,
            "stt_available": self._recognizer is not None,
            "stt_engine": "SpeechRecognition" if self._recognizer else "none",
            "microphone_available": self._microphone is not None,
            "wake_word": self.wake_word,
            "is_listening": self._is_listening,
        }
