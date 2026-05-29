"""
JARVIS-PRIME System Automation
=================================

Controls the Windows desktop:
- Open/close applications
- Web browsing & search
- File operations
- System info & control
- Clipboard & typing
- Volume & brightness
- Screenshot

All actions use built-in Python + Windows APIs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any


# ──────────────────────────────────────────────────────
# Application Registry (common Windows apps)
# ──────────────────────────────────────────────────────

APP_REGISTRY: dict[str, str | list[str]] = {
    # Browsers
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "edge": "msedge",
    "brave": "brave",
    # Microsoft Office
    "word": "winword",
    "excel": "excel",
    "powerpoint": "powerpnt",
    "outlook": "outlook",
    "teams": "msteams",
    "onenote": "onenote",
    # Development
    "vscode": "code",
    "visual studio code": "code",
    "vs code": "code",
    "notepad": "notepad",
    "notepad++": "notepad++",
    "terminal": "wt",
    "windows terminal": "wt",
    "command prompt": "cmd",
    "cmd": "cmd",
    "powershell": "powershell",
    "git bash": "git-bash",
    # System
    "file explorer": "explorer",
    "explorer": "explorer",
    "task manager": "taskmgr",
    "settings": "ms-settings:",
    "control panel": "control",
    "calculator": "calc",
    "paint": "mspaint",
    "snipping tool": "snippingtool",
    "screen recorder": "ms-screenclip:",
    # Media
    "spotify": "spotify",
    "vlc": "vlc",
    "photos": "ms-photos:",
    # Communication
    "discord": "discord",
    "whatsapp": "whatsapp",
    "telegram": "telegram",
    "slack": "slack",
    "zoom": "zoom",
}

# ──────────────────────────────────────────────────────
# System Automation Class
# ──────────────────────────────────────────────────────

class SystemAutomation:
    """
    Controls Windows desktop: apps, browser, files, system.
    """

    def __init__(self):
        self._action_log: list[dict[str, Any]] = []

    def _log(self, action: str, details: str, success: bool = True) -> None:
        self._action_log.append({
            "time": time.strftime("%H:%M:%S"),
            "action": action,
            "details": details,
            "success": success,
        })

    # ─── Application Control ───

    def open_application(self, app_name: str) -> dict[str, Any]:
        """Open an application by name."""
        name_lower = app_name.lower().strip()

        # Check registry
        executable = APP_REGISTRY.get(name_lower)

        if not executable:
            # Try fuzzy match
            for key, exe in APP_REGISTRY.items():
                if name_lower in key or key in name_lower:
                    executable = exe
                    break

        if not executable:
            # Try running directly as a command
            executable = name_lower

        try:
            # Handle URI-style apps (ms-settings:, ms-photos:, etc.)
            if isinstance(executable, str) and executable.endswith(":"):
                os.startfile(executable)
            else:
                subprocess.Popen(
                    executable if isinstance(executable, str) else executable[0],
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            self._log("open_app", f"Opened {app_name} ({executable})")
            return {"status": "success", "app": app_name, "command": str(executable)}

        except Exception as e:
            self._log("open_app", f"Failed to open {app_name}: {e}", False)
            return {"status": "error", "app": app_name, "error": str(e)}

    def close_application(self, app_name: str) -> dict[str, Any]:
        """Close an application by name."""
        name_lower = app_name.lower().strip()
        executable = APP_REGISTRY.get(name_lower, name_lower)

        try:
            if isinstance(executable, str) and not executable.endswith(":"):
                proc_name = executable
                if not proc_name.endswith(".exe"):
                    proc_name += ".exe"
                subprocess.run(
                    ["taskkill", "/IM", proc_name, "/F"],
                    capture_output=True, text=True,
                )
                self._log("close_app", f"Closed {app_name}")
                return {"status": "success", "app": app_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

        return {"status": "error", "error": f"Cannot close {app_name}"}

    # ─── Web Browser ───

    def open_website(self, url: str) -> dict[str, Any]:
        """Open a URL in the default browser."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        webbrowser.open(url)
        self._log("web", f"Opened {url}")
        return {"status": "success", "url": url}

    def google_search(self, query: str) -> dict[str, Any]:
        """Search Google."""
        import urllib.parse
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
        webbrowser.open(url)
        self._log("search", f"Searched: {query}")
        return {"status": "success", "query": query, "url": url}

    def youtube_search(self, query: str) -> dict[str, Any]:
        """Search YouTube."""
        import urllib.parse
        url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
        webbrowser.open(url)
        self._log("youtube", f"YouTube: {query}")
        return {"status": "success", "query": query, "url": url}

    # ─── System Control ───

    def get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        import platform
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()[:60],
            "python": platform.python_version(),
        }

        try:
            import psutil
            mem = psutil.virtual_memory()
            info["cpu_percent"] = psutil.cpu_percent(interval=0.5)
            info["ram_total_gb"] = round(mem.total / 1e9, 1)
            info["ram_used_gb"] = round(mem.used / 1e9, 1)
            info["ram_percent"] = mem.percent
            info["disk_percent"] = psutil.disk_usage('C:\\').percent
            info["battery"] = None
            batt = psutil.sensors_battery()
            if batt:
                info["battery"] = {
                    "percent": batt.percent,
                    "plugged_in": batt.power_plugged,
                }
        except ImportError:
            info["note"] = "Install psutil for detailed system info"

        return info

    def set_volume(self, level: int) -> dict[str, Any]:
        """Set system volume (0-100)."""
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(max(0, min(100, level)) / 100, None)
            return {"status": "success", "volume": level}
        except Exception:
            # Fallback: use nircmd or PowerShell
            try:
                # Use PowerShell approach
                vol = max(0, min(100, level))
                subprocess.run(
                    ['powershell', '-c',
                     f'(New-Object -ComObject WScript.Shell).SendKeys([char]173)'],
                    capture_output=True,
                )
                return {"status": "partial", "volume": level, "note": "Volume key sent"}
            except Exception as e:
                return {"status": "error", "error": str(e)}

    def take_screenshot(self, filename: str = "screenshot.png") -> dict[str, Any]:
        """Take a screenshot."""
        try:
            import pyautogui
            path = Path.cwd() / filename
            pyautogui.screenshot(str(path))
            self._log("screenshot", str(path))
            return {"status": "success", "path": str(path)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def type_text(self, text: str) -> dict[str, Any]:
        """Type text using keyboard simulation."""
        try:
            import pyautogui
            pyautogui.typewrite(text, interval=0.02)
            return {"status": "success", "typed": text[:50]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def press_hotkey(self, *keys: str) -> dict[str, Any]:
        """Press a keyboard hotkey (e.g., 'ctrl', 'c')."""
        try:
            import pyautogui
            pyautogui.hotkey(*keys)
            return {"status": "success", "keys": list(keys)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ─── File Operations ───

    def open_folder(self, path: str) -> dict[str, Any]:
        """Open a folder in File Explorer."""
        target = Path(path)
        if target.exists():
            os.startfile(str(target))
            return {"status": "success", "path": str(target)}
        return {"status": "error", "error": f"Path not found: {path}"}

    def list_files(self, path: str = ".", pattern: str = "*") -> dict[str, Any]:
        """List files in a directory."""
        target = Path(path)
        if not target.exists():
            return {"status": "error", "error": "Path not found"}

        files = list(target.glob(pattern))[:50]
        return {
            "status": "success",
            "path": str(target.absolute()),
            "count": len(files),
            "files": [{"name": f.name, "is_dir": f.is_dir(), "size": f.stat().st_size if f.is_file() else 0} for f in files[:20]],
        }

    # ─── Utility ───

    def get_time(self) -> dict[str, Any]:
        """Get current date and time."""
        now = time.localtime()
        return {
            "time": time.strftime("%I:%M %p", now),
            "date": time.strftime("%A, %B %d, %Y", now),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S", now),
        }

    def get_weather(self, city: str = "Delhi") -> dict[str, Any]:
        """Get weather (opens wttr.in in terminal)."""
        self.open_website(f"https://wttr.in/{city}")
        return {"status": "opened", "city": city, "url": f"https://wttr.in/{city}"}

    def run_command(self, command: str) -> dict[str, Any]:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=15
            )
            return {
                "status": "success",
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "command": command}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def shutdown(self, mode: str = "shutdown", delay: int = 60) -> dict[str, Any]:
        """Shutdown, restart, or sleep the computer."""
        if mode == "shutdown":
            os.system(f"shutdown /s /t {delay}")
        elif mode == "restart":
            os.system(f"shutdown /r /t {delay}")
        elif mode == "sleep":
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        elif mode == "cancel":
            os.system("shutdown /a")
        return {"status": "scheduled", "mode": mode, "delay_seconds": delay}

    def get_action_log(self) -> list[dict[str, Any]]:
        return self._action_log[-20:]
