"""
JARVIS-PRIME Assistant Tests
================================
Tests the assistant brain, intent detection, and automation.
"""
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, success, detail=""):
        if success:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailures:")
            for err in self.errors:
                print(f"  - {err}")
        print(f"{'=' * 60}\n")


def test_intent_detection(results):
    """Test the fast intent detection system."""
    print("\n[Intent Detection Tests]")
    from jarvis.assistant.brain import IntentDetector

    # Open app
    intent, arg = IntentDetector.detect("open chrome")
    results.record("Open app intent", intent == "open_app")
    results.record("Open app arg=chrome", arg == "chrome")

    intent, arg = IntentDetector.detect("launch Visual Studio Code")
    results.record("Launch app intent", intent == "open_app")

    # Close app
    intent, arg = IntentDetector.detect("close notepad")
    results.record("Close app intent", intent == "close_app")

    # Search
    intent, arg = IntentDetector.detect("search for quantum computing")
    results.record("Google search intent", intent == "search_google")
    results.record("Search arg", "quantum computing" in arg)

    intent, arg = IntentDetector.detect("play lofi music on youtube")
    results.record("YouTube search intent", intent == "search_youtube")

    # Time/date
    intent, _ = IntentDetector.detect("what time is it")
    results.record("Time intent", intent == "time")

    intent, _ = IntentDetector.detect("what's the date")
    results.record("Date intent", intent == "date")

    # System
    intent, _ = IntentDetector.detect("system info")
    results.record("System info intent", intent == "system_info")

    intent, _ = IntentDetector.detect("take a screenshot")
    results.record("Screenshot intent", intent == "screenshot")

    # Volume
    intent, arg = IntentDetector.detect("set volume to 50")
    results.record("Volume intent", intent == "volume")
    results.record("Volume arg=50", arg == "50")

    # Greetings
    intent, _ = IntentDetector.detect("hello")
    results.record("Greeting intent", intent == "greeting")

    intent, _ = IntentDetector.detect("how are you")
    results.record("How are you intent", intent == "how_are_you")

    intent, _ = IntentDetector.detect("who are you")
    results.record("Who are you intent", intent == "who_are_you")

    # Website
    intent, _ = IntentDetector.detect("go to github.com")
    results.record("Website intent", intent == "open_website")

    # Conversation fallback
    intent, _ = IntentDetector.detect("explain general relativity to me")
    results.record("Conversation fallback", intent == "conversation")

    # Performance
    start = time.perf_counter()
    for _ in range(1000):
        IntentDetector.detect("open chrome and search for AI")
    elapsed = (time.perf_counter() - start) * 1000
    results.record(f"1000 detections in {elapsed:.0f}ms", elapsed < 100)


def test_automation(results):
    """Test system automation (non-destructive only)."""
    print("\n[System Automation Tests]")
    from jarvis.assistant.automation import SystemAutomation, APP_REGISTRY

    auto = SystemAutomation()

    # App registry
    results.record("App registry has 30+ entries", len(APP_REGISTRY) >= 30)
    results.record("Chrome in registry", "chrome" in APP_REGISTRY)
    results.record("VS Code in registry", "vscode" in APP_REGISTRY)
    results.record("Calculator in registry", "calculator" in APP_REGISTRY)

    # Time
    info = auto.get_time()
    results.record("Get time works", "time" in info)
    results.record("Get date works", "date" in info)

    # System info
    info = auto.get_system_info()
    results.record("System info has OS", "os" in info)
    results.record("System info has Python", "python" in info)

    # File listing
    files = auto.list_files(".", "*.py")
    results.record("File listing works", files["status"] == "success")

    # Run command
    cmd = auto.run_command("echo hello")
    results.record("Run command works", cmd["status"] == "success")
    results.record("Command output correct", "hello" in cmd.get("stdout", ""))

    # Action log
    log = auto.get_action_log()
    results.record("Action log records", isinstance(log, list))


def test_brain(results):
    """Test the brain command processor."""
    print("\n[Brain Processor Tests]")
    from jarvis.assistant.brain import JarvisBrain

    brain = JarvisBrain()

    # Greeting
    response = asyncio.run(brain.process("hello"))
    results.record("Greeting response", "morning" in response.lower() or "afternoon" in response.lower() or "evening" in response.lower())

    # Time
    response = asyncio.run(brain.process("what time is it"))
    results.record("Time response", ":" in response)

    # Who are you
    response = asyncio.run(brain.process("who are you"))
    results.record("Identity response", "jarvis" in response.lower())

    # System info
    response = asyncio.run(brain.process("how is my system"))
    results.record("System response", "%" in response)

    # Thanks
    response = asyncio.run(brain.process("thanks"))
    results.record("Thanks response", len(response) > 5)


def test_ollama_brain(results):
    """Test Ollama LLM connection."""
    print("\n[Ollama LLM Tests]")
    from jarvis.assistant.brain import OllamaBrain

    ollama = OllamaBrain()
    available = asyncio.run(ollama.is_available())
    results.record("Ollama reachable", available)

    if available:
        response = asyncio.run(ollama.chat("Say hello in one word"))
        results.record("Ollama responds", len(response) > 0)
        results.record("Response is text", isinstance(response, str))
    else:
        print("  [SKIP] Ollama not running — skipping LLM tests")


def main():
    print("=" * 60)
    print("  JARVIS-PRIME Assistant Test Suite")
    print("=" * 60)

    results = Results()

    test_intent_detection(results)
    test_automation(results)
    test_brain(results)
    test_ollama_brain(results)

    results.summary()

    if results.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
