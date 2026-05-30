"""
JARVIS-PRIME Desktop Assistant — Main Launcher
==================================================

The real JARVIS. Runs on your desktop with:
- 3D animated avatar (always visible)
- Voice activation ("JARVIS" wake word)
- Natural language understanding via Ollama
- Full desktop automation (open apps, search, etc.)
- Sub-second response times
- Global hotkey: Ctrl+Shift+Q to kill JARVIS instantly

Usage:
    python -m jarvis.assistant.main
    python -m jarvis.assistant.main --text     (text mode, no voice)
    python -m jarvis.assistant.main --no-avatar (voice only, no avatar)
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import sys
import threading
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ──────────────────────────────────────────────────────
# Global Hotkey System (Windows)
# ──────────────────────────────────────────────────────

class HotkeyManager:
    """
    Registers global hotkeys using the Windows API.
    Works even when JARVIS is in the background.

    Default hotkeys:
        Ctrl+Shift+Q  → Kill JARVIS entirely
        Ctrl+Shift+S  → Stop current task / cancel speech
    """

    MOD_CTRL = 0x0002
    MOD_SHIFT = 0x0004
    MOD_ALT = 0x0001

    HOTKEY_KILL = 1
    HOTKEY_STOP = 2

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._running = False
        self.on_kill: callable = None
        self.on_stop: callable = None

    def start(self):
        """Start listening for global hotkeys in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("  [HOTKEY] Ctrl+Shift+Q = Kill JARVIS")
        print("  [HOTKEY] Ctrl+Shift+S = Stop current task")

    def _listen_loop(self):
        try:
            user32 = ctypes.windll.user32

            # Register Ctrl+Shift+Q (Q = 0x51)
            user32.RegisterHotKey(None, self.HOTKEY_KILL,
                                  self.MOD_CTRL | self.MOD_SHIFT, 0x51)
            # Register Ctrl+Shift+S (S = 0x53)
            user32.RegisterHotKey(None, self.HOTKEY_STOP,
                                  self.MOD_CTRL | self.MOD_SHIFT, 0x53)

            msg = ctypes.wintypes.MSG()
            while self._running:
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == 0x0312:  # WM_HOTKEY
                        if msg.wParam == self.HOTKEY_KILL:
                            print("\n  [HOTKEY] Ctrl+Shift+Q pressed — Killing JARVIS!")
                            if self.on_kill:
                                self.on_kill()
                            os._exit(0)
                        elif msg.wParam == self.HOTKEY_STOP:
                            print("\n  [HOTKEY] Ctrl+Shift+S pressed — Stopping task")
                            if self.on_stop:
                                self.on_stop()
                time.sleep(0.05)

            user32.UnregisterHotKey(None, self.HOTKEY_KILL)
            user32.UnregisterHotKey(None, self.HOTKEY_STOP)
        except Exception as e:
            print(f"  [HOTKEY] Could not register hotkeys: {e}")

    def stop(self):
        self._running = False


# We need ctypes.wintypes for MSG struct
try:
    import ctypes.wintypes
except Exception:
    pass


# ──────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────

def run_text_mode():
    """Run JARVIS in text-only mode (no voice, no avatar)."""
    from jarvis.assistant.brain import JarvisBrain

    brain = JarvisBrain()

    print()
    print("=" * 55)
    print("  +---------------------------------------+")
    print("  |   J.A.R.V.I.S. — PRIME ASSISTANT      |")
    print("  |   Text Mode | Type 'quit' to exit     |")
    print("  +---------------------------------------+")
    print("=" * 55)
    print()

    # Quick check Ollama
    available = asyncio.run(brain.llm.is_available())
    if available:
        print("  ✓ Ollama connected | Model: " + brain.llm.model)
    else:
        print("  ✗ Ollama not running — start with: ollama serve")
        print("    (Will use pattern matching for commands)")
    print()

    # Hotkeys
    hotkeys = HotkeyManager()
    hotkeys.on_kill = lambda: os._exit(0)
    hotkeys.start()

    while True:
        try:
            user_input = input("  You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
                print("\n  JARVIS: Goodbye! Have a great day.\n")
                break

            response, _ = asyncio.run(brain.process(user_input))
            print(f"  JARVIS: {response}\n")

        except KeyboardInterrupt:
            print("\n\n  JARVIS: Shutting down. Goodbye!\n")
            break
        except EOFError:
            break


def run_voice_mode():
    """Run JARVIS with voice (no avatar window)."""
    from jarvis.assistant.brain import JarvisBrain
    from jarvis.assistant.voice import VoiceEngine

    brain = JarvisBrain()
    voice = VoiceEngine(wake_word="jarvis")

    print()
    print("=" * 55)
    print("  +---------------------------------------+")
    print('  |   J.A.R.V.I.S. — VOICE MODE           |')
    print('  |   Say "JARVIS" to activate            |')
    print("  +---------------------------------------+")
    print("=" * 55)
    print()

    # Hotkeys
    hotkeys = HotkeyManager()
    hotkeys.on_kill = lambda: os._exit(0)
    hotkeys.on_stop = lambda: voice.stop_listening()
    hotkeys.start()

    def on_wake():
        print("  [WAKE] Detected!")
        voice.speak("Yes?", block=True)

    def on_command(command: str):
        # "stop" command
        if command.lower().strip() in ("stop", "shut up", "cancel", "be quiet", "nevermind"):
            print("  JARVIS: Okay, standing by.")
            voice.speak("Standing by.", block=True)
            return

        print(f"  You: {command}")
        response, _ = asyncio.run(brain.process(command))
        print(f"  JARVIS: {response}")
        voice.speak(response, block=True)

    voice.start_wake_word_listener(on_wake=on_wake, on_command=on_command)

    print("  Listening for wake word... (Ctrl+Shift+Q to kill)\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        voice.stop_listening()
        print("\n  JARVIS: Goodbye!\n")


def run_avatar_mode():
    """Run JARVIS with the 3D desktop pet (PyQt6) + voice + vision."""
    from jarvis.assistant.brain import JarvisBrain

    brain = JarvisBrain()

    # ─── Hotkeys ───
    hotkeys = HotkeyManager()
    _stop_flag = threading.Event()

    def kill_all():
        print("\n  [JARVIS] Emergency shutdown via hotkey.")
        os._exit(0)

    def stop_current():
        _stop_flag.set()
        print("  [JARVIS] Current task cancelled.")

    hotkeys.on_kill = kill_all
    hotkeys.on_stop = stop_current
    hotkeys.start()

    # ─── Voice ───
    voice_engine = None
    try:
        from jarvis.assistant.voice import VoiceEngine
        voice_engine = VoiceEngine(wake_word="jarvis")
    except Exception as e:
        print(f"[WARN] Voice unavailable: {e}")

    # ─── Biometrics (startup-only, brief scan) ───
    try:
        from jarvis.perception.biometrics import VisionBiometrics
        vision = VisionBiometrics()

        def on_user_detected(name):
            if voice_engine:
                voice_engine.speak_async(f"Welcome back, {name}.")
            print(f"  [JARVIS] Identified user: {name}")

        vision.on_user_detected = on_user_detected

        # Run the brief startup scan in a thread so it doesn't block
        scan_thread = threading.Thread(
            target=vision.identify_at_startup,
            kwargs={"max_seconds": 5.0},
            daemon=True,
        )
        scan_thread.start()
    except Exception as e:
        print(f"  [WARN] Vision biometrics unavailable: {e}")

    # ─── Desktop Pet ───
    pet_controller = None
    try:
        from jarvis.assistant.desktop_pet import PetController
        pet_controller = PetController()
        print("  [JARVIS] Desktop Pet initializing on main thread...")
    except Exception as e:
        print(f"  [ERROR] Could not initialize desktop pet: {e}")
        print("          Falling back to text mode...\n")
        run_text_mode()
        return

    # ─── Wake word listener ───
    if voice_engine:
        def on_wake():
            _stop_flag.clear()
            print("  [JARVIS]: Yes? I'm listening...")
            if pet_controller:
                pet_controller.set_state("listening")
                pet_controller.speak("Yes? I'm listening...")
            voice_engine.speak("Yes?", block=True)

        def on_command(command: str):
            # Stop commands
            cmd_lower = command.lower().strip()
            if cmd_lower in ("stop", "shut up", "cancel", "be quiet",
                             "nevermind", "never mind", "that's enough"):
                if pet_controller:
                    pet_controller.set_state("")
                    pet_controller.speak("Standing by.")
                voice_engine.speak("Standing by.", block=True)
                return

            _stop_flag.clear()

            print(f"  [YOU]: {command}")

            if pet_controller:
                pet_controller.set_state("thinking")
                pet_controller.speak(f'"{command}"')
            time.sleep(0.3)

            # Check stop flag before processing
            if _stop_flag.is_set():
                return

            response, avatar_action = asyncio.run(brain.process(command))

            # Check stop flag before speaking
            if _stop_flag.is_set():
                return

            print(f"  [JARVIS]: {response}")

            if pet_controller:
                # Physical commands driven by LLM tools
                if avatar_action == "jump":
                    pet_controller.jump()
                elif avatar_action == "move_left":
                    pet_controller.move_left()
                elif avatar_action == "move_right":
                    pet_controller.move_right()
                elif avatar_action == "scale_up":
                    pet_controller.scale(1.2)
                elif avatar_action == "scale_down":
                    pet_controller.scale(0.8)
                elif avatar_action == "hide":
                    pet_controller.hide()
                elif avatar_action == "show":
                    pet_controller.show()
                elif avatar_action == "shake":
                    pet_controller.shake()

                # Dynamic expression parsing based on response text
                expr = ""
                lower_resp = response.lower()
                if any(w in lower_resp for w in ["haha", "joke", "funny", "happy", "great", "excellent"]):
                    expr = "happy"
                elif any(w in lower_resp for w in ["cute", "love", "sweet", "aww", "sir", "welcome"]):
                    expr = "cute"
                
                pet_controller.set_expression(expr)
                pet_controller.set_state("speaking")
                pet_controller.speak(response)

            voice_engine.speak(response, block=True)
            if pet_controller:
                pet_controller.set_state("")
                pet_controller.set_expression("")

        voice_engine.start_wake_word_listener(on_wake=on_wake, on_command=on_command)

        def on_text_command(text: str):
            import threading
            # Run the command loop in a new thread so it doesn't freeze the GUI
            threading.Thread(target=on_command, args=(text,), daemon=True).start()

        if pet_controller:
            pet_controller.text_command_callback = on_text_command
    # ─── Launch ───
    print()
    print("=" * 55)
    print("  +---------------------------------------+")
    print("  |   J.A.R.V.I.S. — DESKTOP ENTITY       |")
    print("  |   Living Avatar + Voice + Vision      |")
    print("  +---------------------------------------+")
    print("=" * 55)
    print()
    print('  Say "JARVIS" to interact')
    print("  Ctrl+Shift+Q = Kill | Ctrl+Shift+S = Stop task")
    print()

    try:
        if pet_controller:
            # This blocks the main thread and processes PyQt events
            pet_controller.start_sync()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        if voice_engine:
            voice_engine.stop_listening()
        hotkeys.stop()
        print("\n  JARVIS: Shutting down. Goodbye!\n")
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="JARVIS-PRIME Desktop Assistant")
    parser.add_argument("--text", action="store_true", help="Text-only mode (no voice, no avatar)")
    parser.add_argument("--voice", action="store_true", help="Voice-only mode (no avatar)")
    parser.add_argument("--no-avatar", action="store_true", help="Same as --voice")
    parser.add_argument("--install-autostart", action="store_true", help="Enable Windows auto-start")
    parser.add_argument("--remove-autostart", action="store_true", help="Disable Windows auto-start")
    parser.add_argument("--model", default="qwen3:1.7b", help="Ollama model (default: qwen3:1.7b)")
    args = parser.parse_args()

    if args.install_autostart:
        from jarvis.infrastructure.startup_manager import StartupManager
        StartupManager.enable_autostart()
        print("Auto-start installed. Exiting.")
        return

    if args.remove_autostart:
        from jarvis.infrastructure.startup_manager import StartupManager
        StartupManager.disable_autostart()
        print("Auto-start removed. Exiting.")
        return

    if args.text:
        run_text_mode()
    elif args.voice or args.no_avatar:
        run_voice_mode()
    else:
        run_avatar_mode()


if __name__ == "__main__":
    main()
