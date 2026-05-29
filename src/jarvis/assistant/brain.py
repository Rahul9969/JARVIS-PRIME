"""
JARVIS-PRIME Brain — Natural Language Command Processor
==========================================================

Takes natural language input and routes to the correct action:
1. Understands intent (open app, search, question, system control, etc.)
2. Routes to SystemAutomation for actions
3. Routes to Ollama/LLM for conversational responses
4. Maintains conversation context

Uses Ollama locally for sub-second responses.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

import httpx

from jarvis.assistant.automation import SystemAutomation


# ──────────────────────────────────────────────────────
# Intent Detection (keyword + pattern matching)
# ──────────────────────────────────────────────────────

class IntentDetector:
    """
    Fast keyword-based intent detection.
    Runs in <1ms — no LLM needed for common commands.
    """

    PATTERNS = {
        "open_app": [
            r"(?:open|launch|start|run)\s+(.+)",
        ],
        "close_app": [
            r"(?:close|quit|exit|kill|stop)\s+(.+)",
        ],
        "search_google": [
            r"(?:search|google|look up|find)\s+(?:for\s+)?(.+)",
        ],
        "search_youtube": [
            r"(?:play|youtube|watch)\s+(.+)",
        ],
        "open_website": [
            r"(?:go to|visit|open|browse)\s+((?:https?://)?[\w.-]+\.\w+.*)",
        ],
        "time": [
            r"what(?:'s| is) the time",
            r"what time is it",
            r"current time",
            r"tell me the time",
        ],
        "date": [
            r"what(?:'s| is) (?:the |today(?:'s)? )?date",
            r"what day is it",
        ],
        "system_info": [
            r"(?:system|cpu|ram|memory|battery|disk)\s*(?:info|status|usage|level)?",
            r"how(?:'s| is) (?:the |my )?(?:system|computer|pc|laptop)",
        ],
        "screenshot": [
            r"(?:take|capture)\s+(?:a\s+)?screenshot",
            r"screenshot",
        ],
        "volume": [
            r"(?:set|change)\s+volume\s+(?:to\s+)?(\d+)",
            r"volume\s+(\d+)",
        ],
        "shutdown": [
            r"shutdown|shut down",
        ],
        "restart": [
            r"restart|reboot",
        ],
        "sleep": [
            r"(?:go to |)sleep|hibernate",
        ],
        "weather": [
            r"weather\s*(?:in\s+)?(.+)?",
        ],
        "greeting": [
            r"^(?:hi|hello|hey|good (?:morning|afternoon|evening)|what's up)$",
        ],
        "thanks": [
            r"^(?:thanks?|thank you|good job|great|awesome|nice).*$",
        ],
        "how_are_you": [
            r"how are you",
            r"how(?:'re| are) you doing",
        ],
        "who_are_you": [
            r"who are you",
            r"what(?:'s| is) your name",
            r"tell me about yourself",
            r"what can you do",
        ],
        "stop": [
            r"^(?:stop|shut up|cancel|be quiet|nevermind|never mind|that's enough)$",
        ],
        "self_improvement": [
            r"(?:build|create|write|add)\s+(?:a\s+)?(?:new\s+)?(?:tool|feature|capability)\s+(?:to|for|that)\s+(.+)",
            r"(?:improve|optimize|rewrite)\s+(?:your\s+)?(.+)",
            r"(?:teach\s+yourself\s+how\s+to|learn\s+how\s+to)\s+(.+)",
        ],
    }

    @classmethod
    def detect(cls, text: str) -> tuple[str, str]:
        """
        Detect intent from text.
        Returns (intent, extracted_argument).
        """
        text_lower = text.lower().strip()

        for intent, patterns in cls.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    arg = match.group(1).strip() if match.lastindex else ""
                    return intent, arg

        return "conversation", text


# ──────────────────────────────────────────────────────
# Ollama LLM Interface
# ──────────────────────────────────────────────────────

class OllamaBrain:
    """
    Direct Ollama API interface for conversational AI.
    Sub-second responses with small models.
    """

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
    ):
        self.model = model
        self.base_url = base_url
        self.conversation: list[dict[str, str]] = []
        self.system_prompt = (
            "You are JARVIS, a highly intelligent, witty, and helpful AI assistant. "
            "You speak naturally and conversationally, like a brilliant friend. "
            "Keep responses concise (1-3 sentences) unless asked for detail. "
            "You can be playful and have personality. You call the user 'Sir' occasionally. "
            "Never mention that you're an AI language model — you ARE JARVIS."
        )

    async def chat(self, message: str) -> str:
        """Send a message to Ollama and get a response."""
        self.conversation.append({"role": "user", "content": message})

        # Keep context window manageable
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-16:]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            *self.conversation,
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 150,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("message", {}).get("content", "").strip()
                    # Strip think tags if present (qwen3 thinking mode)
                    reply = re.sub(r'<think>.*?</think>', '', reply, flags=re.DOTALL).strip()
                    if reply:
                        self.conversation.append({"role": "assistant", "content": reply})
                        return reply

        except httpx.ConnectError:
            return "I can't reach my brain right now. Make sure Ollama is running with: ollama serve"
        except Exception as e:
            return f"Something went wrong: {str(e)[:100]}"

        return "I'm not sure how to respond to that."

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False


# ──────────────────────────────────────────────────────
# Main Brain — Command Processor
# ──────────────────────────────────────────────────────

class JarvisBrain:
    """
    The brain that processes all commands.
    Routes between automation and conversation.
    """

    def __init__(self, ollama_model: str = "qwen3:1.7b"):
        self.automation = SystemAutomation()
        self.llm = OllamaBrain(model=ollama_model)
        self.intent_detector = IntentDetector()
        self._start_time = time.time()

    async def process(self, text: str) -> str:
        """
        Process a natural language command.
        Returns the response text to speak.
        """
        if not text:
            return "I didn't catch that. Could you repeat?"

        intent, arg = self.intent_detector.detect(text)

        # ─── Action intents ───
        if intent == "open_app":
            result = self.automation.open_application(arg)
            if result["status"] == "success":
                return f"Opening {arg} for you."
            return f"Sorry, I couldn't open {arg}."

        elif intent == "close_app":
            result = self.automation.close_application(arg)
            return f"Closing {arg}." if result["status"] == "success" else f"Couldn't close {arg}."

        elif intent == "search_google":
            self.automation.google_search(arg)
            return f"Searching for {arg}."

        elif intent == "search_youtube":
            self.automation.youtube_search(arg)
            return f"Playing {arg} on YouTube."

        elif intent == "open_website":
            self.automation.open_website(arg)
            return f"Opening {arg}."

        elif intent == "time":
            info = self.automation.get_time()
            return f"It's {info['time']}."

        elif intent == "date":
            info = self.automation.get_time()
            return f"Today is {info['date']}."

        elif intent == "system_info":
            info = self.automation.get_system_info()
            cpu = info.get('cpu_percent', '?')
            ram = info.get('ram_percent', '?')
            return f"CPU is at {cpu}%, RAM at {ram}%. Everything looks normal."

        elif intent == "screenshot":
            result = self.automation.take_screenshot()
            return "Screenshot taken!" if result["status"] == "success" else "Couldn't take screenshot."

        elif intent == "volume":
            try:
                level = int(arg)
                self.automation.set_volume(level)
                return f"Volume set to {level}%."
            except ValueError:
                return "What volume level? Say a number from 0 to 100."

        elif intent == "shutdown":
            return "Initiating shutdown in 60 seconds. Say 'cancel shutdown' to abort."

        elif intent == "restart":
            return "Restarting in 60 seconds."

        elif intent == "weather":
            city = arg if arg else "Delhi"
            self.automation.get_weather(city)
            return f"Opening weather for {city}."

        elif intent == "greeting":
            hour = time.localtime().tm_hour
            if hour < 12:
                return "Good morning! How can I help you today?"
            elif hour < 17:
                return "Good afternoon! What can I do for you?"
            else:
                return "Good evening! What do you need?"

        elif intent == "thanks":
            return "You're welcome! Always here to help."

        elif intent == "how_are_you":
            return "I'm running perfectly. All systems operational. What can I do for you?"

        elif intent == "who_are_you":
            return (
                "I'm JARVIS, your personal AI assistant. "
                "I can open apps, search the web, answer questions, "
                "control your system, and much more. Just ask!"
            )
            
        elif intent == "stop":
            return "Standing by."

        elif intent == "self_improvement":
            # Avoid blocking the main event loop too long by wrapping in a task if possible, 
            # or just return a response that it's starting, but for now we'll await it.
            try:
                from jarvis.core.metacognition import SelfImprovementAgent
                agent = SelfImprovementAgent(llm_model=self.llm.model)
                
                # We need to decide which file to target. For simplicity, if it's a new tool, 
                # we'll target a generic extensions file or ask the agent to create one.
                # Let's put new tools in jarvis.assistant.automation for now.
                target = "src/jarvis/assistant/automation.py"
                
                result = await agent.attempt_improvement(target, arg)
                if result["status"] == "success":
                    return "I have successfully written and injected the new capability into my brain."
                else:
                    return f"I encountered an error while trying to improve myself: {result['message']}"
            except Exception as e:
                return f"Self-improvement failed: {e}"

        # ─── Conversation (LLM) ───
        else:
            return await self.llm.chat(text)
