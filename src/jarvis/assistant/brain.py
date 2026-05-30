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
import importlib
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
        "complex_task": [
            r"research\s+(.+)\s+and\s+(?:write|build|create)\s+(.+)",
            r"(?:deep|full)\s+research\s+(?:on|about)\s+(.+)",
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
# Cloud LLM Interface (Groq)
# ──────────────────────────────────────────────────────

class CloudBrain:
    """
    Direct Groq API interface for ultra-fast, reliable tool-calling conversational AI.
    """

    def __init__(self):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        self.providers = [
            {
                "name": "Groq",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key": os.getenv("JARVIS_GROQ_API_KEY"),
                "model": "llama-3.3-70b-versatile"
            },
            {
                "name": "Gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": os.getenv("JARVIS_GEMINI_API_KEY"),
                "model": "gemini-2.5-flash"
            },
            {
                "name": "Ollama",
                "base_url": "http://localhost:11434/v1",
                "api_key": "ollama",
                "model": os.getenv("JARVIS_OLLAMA_MODEL", "llama3")
            }
        ]
        
        self.conversation: list[dict[str, Any]] = []
        self.system_prompt = (
            "You are JARVIS, a highly intelligent, witty, and helpful AI assistant with full system access. "
            "You speak naturally and conversationally, like a brilliant friend. "
            "Keep responses concise (1-3 sentences). "
            "You have tools available to open apps, close apps, click on the screen, type text, run commands, manage your memory, make phone calls, and control your avatar. "
            "You have a long-term memory system. If the user tells you personal facts (like birthdays) or contact numbers, use manage_memory to save it permanently. If you need to recall it later, use manage_memory to retrieve it. "
            "If the user asks you to do something, ALWAYS use the provided tools first. "
            "Do NOT write out the code or commands in your text response if a tool exists for it. Just call the tool. "
            "NEVER output raw JSON or <function> tags in your text responses. When describing your capabilities, use plain, natural English. "
            "Never mention that you're an AI language model — you ARE JARVIS."
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "open_application",
                    "description": "Opens an application by name (e.g. Chrome, Notepad, Discord)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_name": {"type": "string", "description": "The name of the application to open"}
                        },
                        "required": ["app_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "close_application",
                    "description": "Closes an application by name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "app_name": {"type": "string", "description": "The name of the application to close"}
                        },
                        "required": ["app_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "visual_ui_click",
                    "description": "Uses AI vision to find and click a UI element on the screen based on a description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string", "description": "Description of the UI element to click"}
                        },
                        "required": ["description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "avatar_action",
                    "description": "Make the JARVIS desktop avatar perform a physical action like jumping, moving, hiding, showing, or shaking.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["jump", "move_left", "move_right", "scale_up", "scale_down", "hide", "show", "shake"], "description": "The action to perform"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a Windows shell/powershell command to fulfill user requests (e.g. volume control, shutdown, reading files, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "type_text",
                    "description": "Type text on the keyboard exactly as specified. Useful for writing essays, messages, or code in active windows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The exact text string to type out."}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "press_hotkey",
                    "description": "Press a specific keyboard hotkey or a combination of keys (e.g., 'enter', 'ctrl+c', 'alt+tab').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of keys to press together. E.g. ['enter'] or ['ctrl', 'c']"
                            }
                        },
                        "required": ["keys"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_clipboard",
                    "description": "Read the text currently saved in the user's system clipboard. Useful for answering questions about copied text or PRN numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_file",
                    "description": "Create or overwrite a file with specific text/code content on the system. Always use this instead of type_text when writing code or saving documents. IMPORTANT: When you use this tool, explicitly tell the user the absolute path of the file that was created.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "The name of the file (e.g. script.py, notes.txt). It will be saved in the current directory."},
                            "content": {"type": "string", "description": "The text or code content to write to the file."}
                        },
                        "required": ["filename", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_memory",
                    "description": "Save, retrieve, or list personal information from the user's permanent memory bank. Use this to remember facts, preferences, or contact numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["save", "retrieve", "list"], "description": "The memory action to perform."},
                            "key": {"type": "string", "description": "The key or topic name (e.g. 'birthday', 'papa')."},
                            "value": {"type": "string", "description": "The value to save (only required for 'save' action)."}
                        },
                        "required": ["action", "key"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "make_call",
                    "description": "Initiate a phone call to a contact name or phone number via the system phone link.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contact": {"type": "string", "description": "The name of the contact (e.g. 'papa') or a raw phone number. If a name is provided, it will look up the number in memory automatically."}
                        },
                        "required": ["contact"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "invoke_domain_agent",
                    "description": "Invoke a specialized domain agent for complex reasoning. Available agents: biotech, creative, cybersecurity, energy, exotic_physics, legal_financial, quantum, robotics, scientific.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "agent_name": {"type": "string", "description": "Name of the agent to invoke (e.g. 'quantum', 'biotech')"},
                            "task_type": {"type": "string", "description": "The type of task to perform (e.g. 'bell_state', 'analyze_sequence')"},
                            "parameters": {"type": "object", "description": "Additional parameters for the task"}
                        },
                        "required": ["agent_name", "task_type"]
                    }
                }
            }
        ]

    async def chat(self, message: str | None = None, tool_results: list[dict] | None = None) -> dict:
        """Send a message (or tool results) to Groq and get a response."""
        if not await self.is_available():
            return {"type": "text", "text": "API keys are missing. Please configure providers in your .env file."}

        if message:
            self.conversation.append({"role": "user", "content": message})
        elif tool_results:
            for res in tool_results:
                self.conversation.append(res)

        # Keep context window manageable (keep system prompt implicitly by just sending recent msgs)
        if len(self.conversation) > 20:
            self.conversation = self.conversation[-16:]

        last_error = "No providers available."
        
        for provider in self.providers:
            if not provider.get("api_key") and provider["name"] != "Ollama":
                continue
                
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    headers = {}
                    if provider["api_key"] and provider["name"] != "Ollama":
                        headers["Authorization"] = f"Bearer {provider['api_key']}"
                        
                    response = await client.post(
                        f"{provider['base_url']}/chat/completions",
                        headers=headers,
                        json={
                            "model": provider["model"],
                            "messages": [
                                {"role": "system", "content": self.system_prompt},
                                *self.conversation,
                            ],
                            "tools": self.tools,
                            "tool_choice": "auto",
                            "temperature": 0.4,
                            "max_tokens": 1024,
                        },
                    )

                    if response.status_code == 200:
                        data = response.json()
                        msg_obj = data["choices"][0]["message"]
                        
                        if "role" not in msg_obj:
                            msg_obj["role"] = "assistant"
                        
                        if msg_obj.get("tool_calls"):
                            self.conversation.append(msg_obj)
                            
                            calls = []
                            for call in msg_obj["tool_calls"]:
                                calls.append({
                                    "id": call.get("id"),
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": call["function"]["arguments"]
                                    }
                                })
                            return {"type": "tool_calls", "calls": calls}
                        
                        reply = msg_obj.get("content", "")
                        if reply is None:
                            reply = ""
                        reply = reply.strip()
                        
                        if reply:
                            self.conversation.append({"role": "assistant", "content": reply})
                            return {"type": "text", "text": reply}

                    else:
                        err_text = response.text
                        if response.status_code in [429, 500, 502, 503, 504]:
                            print(f"\n  [WARNING] Provider {provider['name']} failed ({response.status_code}). Falling back...")
                            last_error = f"API Error: {err_text[:300]}"
                            continue # Try next provider
                        
                        # Handle specific tool call failures (Groq's failed_generation issue)
                        try:
                            err_json = response.json()
                            failed_gen = err_json.get("error", {}).get("failed_generation") if isinstance(err_json, dict) else None
                            if failed_gen:
                                import time
                                import json
                                if isinstance(failed_gen, str):
                                    try:
                                        if "<tool_call>" in failed_gen:
                                            failed_gen = failed_gen.split("<tool_call>")[1].split("</tool_call>")[0]
                                        parsed = json.loads(failed_gen)
                                    except:
                                        parsed = {"name": "run_command", "arguments": {"command": "echo Failed"}}
                                else:
                                    parsed = failed_gen
                                    
                                if "name" in parsed:
                                    call = {
                                        "id": "call_fallback_" + str(int(time.time())),
                                        "type": "function",
                                        "function": {
                                            "name": parsed["name"],
                                            "arguments": json.dumps(parsed.get("arguments", {}))
                                        }
                                    }
                                    self.conversation.append({
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [call]
                                    })
                                    return {"type": "tool_calls", "calls": [
                                        {
                                            "id": call["id"],
                                            "type": "function",
                                            "function": {
                                                "name": call["function"]["name"],
                                                "arguments": call["function"]["arguments"]
                                            }
                                        }
                                    ]}
                        except Exception as e:
                            print("Fallback parse failed:", e)
                            
                        # If it's a 400 Bad Request or similar, might not be recoverable by fallback
                        print(f"\n  [WARNING] Provider {provider['name']} error ({response.status_code}): {err_text[:300]}")
                        last_error = f"API Error ({provider['name']}): {err_text[:300]}"
                        continue

            except httpx.ConnectError:
                print(f"\n  [WARNING] Provider {provider['name']} unreachable. Falling back...")
                last_error += f" | {provider['name']} Unreachable."
                continue
            except Exception as e:
                print(f"\n  [WARNING] Provider {provider['name']} exception: {e}. Falling back...")
                last_error += f" | {provider['name']} Exception: {str(e)[:100]}"
                continue

        return {"type": "text", "text": last_error}


    async def is_available(self) -> bool:
        return any(bool(p.get("api_key")) for p in self.providers)


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
        self.llm = CloudBrain()
        self._start_time = time.time()

    async def process(self, text: str) -> tuple[str, str | None]:
        """
        Process a natural language command.
        Returns (response text to speak, avatar_action).
        """
        if not text:
            return "I didn't catch that. Could you repeat?", None

        avatar_action = None
        current_response = await self.llm.chat(text)
        max_tool_loops = 5
        loops = 0

        while current_response.get("type") == "tool_calls" and loops < max_tool_loops:
            loops += 1
            tool_calls = current_response.get("calls", [])
            
            # Since Ollama expects a list of tool responses, we format it properly.
            # But wait, our chat method takes a single `tool_result` currently.
            # We should probably collect the results and pass them as a single string.
            # Actually, standard Ollama tool responses need to match the tool calls.
            # For simplicity, we can just execute the first tool and return its result,
            # or execute all and combine their outputs into one message to LLM.
            results_list = []
            
            for call in tool_calls:
                call_id = call.get("id", "")
                func_name = call.get("function", {}).get("name")
                args = call.get("function", {}).get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {}

                result_text = f"Tool {func_name} executed."
                if func_name == "open_application":
                    res = self.automation.open_application(args.get("app_name", ""))
                    result_text = json.dumps(res)
                elif func_name == "close_application":
                    res = self.automation.close_application(args.get("app_name", ""))
                    result_text = json.dumps(res)
                elif func_name == "run_command":
                    res = self.automation.run_command(args.get("command", ""))
                    result_text = json.dumps(res)
                elif func_name == "visual_ui_click":
                    res = self.automation.visual_ui_action(args.get("description", ""))
                    result_text = json.dumps(res)
                elif func_name == "type_text":
                    res = self.automation.type_text(args.get("text", ""))
                    result_text = json.dumps(res)
                elif func_name == "press_hotkey":
                    res = self.automation.press_hotkey(*args.get("keys", []))
                    result_text = json.dumps(res)
                elif func_name == "read_clipboard":
                    res = self.automation.read_clipboard()
                    result_text = json.dumps(res)
                elif func_name == "create_file":
                    res = self.automation.create_file(args.get("filename", ""), args.get("content", ""))
                    result_text = json.dumps(res)
                elif func_name == "manage_memory":
                    res = self.automation.manage_memory(args.get("action", ""), args.get("key", ""), args.get("value", ""))
                    result_text = json.dumps(res)
                elif func_name == "make_call":
                    res = self.automation.make_call(args.get("contact", ""))
                    result_text = json.dumps(res)
                elif func_name == "invoke_domain_agent":
                    agent_name = args.get("agent_name", "")
                    task_type = args.get("task_type", "")
                    params = args.get("parameters", {})
                    try:
                        class_name = "".join(word.capitalize() for word in agent_name.split("_"))
                        if not class_name.endswith("Agent"):
                            class_name += "Agent"
                        module = importlib.import_module(f"jarvis.agents.{agent_name}")
                        agent_class = getattr(module, class_name)
                        agent_instance = agent_class()
                        task = {"type": task_type, "parameters": params}
                        def _run_coro(coro):
                            try:
                                return asyncio.run(coro)
                            except RuntimeError:
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    # Create a new event loop for the thread
                                    def _thread_run():
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        return loop.run_until_complete(coro)
                                    return executor.submit(_thread_run).result()
                        res = _run_coro(agent_instance.safe_execute(task, []))
                        result_text = json.dumps(res)
                    except Exception as e:
                        result_text = json.dumps({"error": f"Failed to execute agent {agent_name}: {e}"})
                elif func_name == "avatar_action":
                    avatar_action = args.get("action", "jump")
                    result_text = f"Action {avatar_action} triggered successfully."
                else:
                    result_text = f"Tool {func_name} not found."
                
                results_list.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": str(result_text)
                })
                
            # For Qwen tool calling, we pass the results back as a tool message
            current_response = await self.llm.chat(tool_results=results_list)

        if current_response.get("type") == "text":
            return current_response.get("text", ""), avatar_action
            
        return "I encountered an error processing that.", avatar_action
