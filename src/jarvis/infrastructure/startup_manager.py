"""
JARVIS-PRIME Startup Manager
=============================

Handles Windows registry keys to ensure JARVIS starts automatically
when the computer boots / user logs in.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None


class StartupManager:
    """Manages Windows auto-start for JARVIS."""
    
    REG_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"
    APP_NAME = "JarvisPrimeAssistant"
    
    @classmethod
    def get_launch_command(cls) -> str:
        """Get the command to launch JARVIS without opening a visible console."""
        pythonw_exe = sys.executable.replace("python.exe", "pythonw.exe")
        if not os.path.exists(pythonw_exe):
            pythonw_exe = sys.executable # fallback
            
        main_script = Path(__file__).parent.parent.parent.parent / "src" / "jarvis" / "assistant" / "main.py"
        
        # We want to launch the desktop pet by default on startup
        return f'"{pythonw_exe}" "{main_script.absolute()}"'

    @classmethod
    def enable_autostart(cls) -> bool:
        """Enable JARVIS to start on Windows boot."""
        if winreg is None:
            return False
            
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                cls.REG_PATH, 
                0, 
                winreg.KEY_SET_VALUE
            )
            winreg.SetValueEx(
                key, 
                cls.APP_NAME, 
                0, 
                winreg.REG_SZ, 
                cls.get_launch_command()
            )
            winreg.CloseKey(key)
            print("[SYSTEM] Auto-start enabled in Windows Registry.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to enable auto-start: {e}")
            return False

    @classmethod
    def disable_autostart(cls) -> bool:
        """Disable JARVIS auto-start."""
        if winreg is None:
            return False
            
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                cls.REG_PATH, 
                0, 
                winreg.KEY_SET_VALUE
            )
            winreg.DeleteValue(key, cls.APP_NAME)
            winreg.CloseKey(key)
            print("[SYSTEM] Auto-start disabled.")
            return True
        except FileNotFoundError:
            return True # Already disabled
        except Exception as e:
            print(f"[ERROR] Failed to disable auto-start: {e}")
            return False

    @classmethod
    def is_autostart_enabled(cls) -> bool:
        """Check if auto-start is currently enabled."""
        if winreg is None:
            return False
            
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, 
                cls.REG_PATH, 
                0, 
                winreg.KEY_READ
            )
            value, _ = winreg.QueryValueEx(key, cls.APP_NAME)
            winreg.CloseKey(key)
            return value == cls.get_launch_command()
        except FileNotFoundError:
            return False
        except Exception:
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--disable", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    
    if args.enable:
        StartupManager.enable_autostart()
    elif args.disable:
        StartupManager.disable_autostart()
    else:
        status = "ENABLED" if StartupManager.is_autostart_enabled() else "DISABLED"
        print(f"JARVIS Auto-start is currently: {status}")
        print(f"Launch command: {StartupManager.get_launch_command()}")
