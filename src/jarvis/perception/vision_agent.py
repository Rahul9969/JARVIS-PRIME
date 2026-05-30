"""
JARVIS-PRIME Vision-Action Agent
================================

Takes screenshots, uses Gemini 1.5 Flash/Pro to locate UI elements via 
2D spatial bounding boxes, and calculates exact screen coordinates for PyAutoGUI.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

try:
    import pyautogui
    from PIL import Image
    from google import genai
    from google.genai import types
except ImportError:
    print("[WARN] Missing dependencies for VisionAgent. Run: pip install google-genai pillow pyautogui")


class VisionAgent:
    """Uses a Vision-Language Model to "see" the screen and act."""

    def __init__(self, api_key: str | None = None):
        # Load key from env if not provided
        if not api_key:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("JARVIS_GEMINI_API_KEY")

        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            print("[WARN] JARVIS_GEMINI_API_KEY not found. Vision Agent disabled.")

        self.model = "gemini-2.5-flash"
        
        # Determine screen resolution
        try:
            self.screen_width, self.screen_height = pyautogui.size()
        except Exception:
            self.screen_width, self.screen_height = 1920, 1080

    def capture_screen(self, save_path: str = "temp_screenshot.png") -> str:
        """Capture the current screen and return the file path."""
        screenshot = pyautogui.screenshot()
        screenshot.save(save_path)
        return save_path

    def locate_element(self, element_description: str) -> tuple[int, int] | None:
        """
        Takes a screenshot, asks Gemini for the bounding box of the element,
        and returns the (x, y) center pixel coordinates.
        """
        if not self.client:
            print("[ERROR] Vision Agent requires JARVIS_GEMINI_API_KEY.")
            return None

        img_path = self.capture_screen()
        
        # Gemini spatial prompt to return bounding box
        prompt = (
            f"Locate the UI element matching this description: '{element_description}'.\n"
            "Return ONLY a JSON array containing the bounding box in this format: [ymin, xmin, ymax, xmax]. "
            "These coordinates should be normalized from 0 to 1000. "
            "If the element is not found, return an empty array []."
        )
        
        print(f"  [VISION] Scanning screen for '{element_description}'...")
        
        try:
            # Upload file to Gemini using the new SDK
            image_file = self.client.files.upload(file=img_path)
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=[image_file, prompt],
            )
            
            # Clean up the remote file
            self.client.files.delete(name=image_file.name)
            os.remove(img_path)
            
            result_text = response.text.strip()
            
            # Extract array
            match = re.search(r"\[([\d\s,]+)\]", result_text)
            if not match:
                print(f"  [VISION] Could not parse bounding box from: {result_text}")
                return None
                
            coords = json.loads(f"[{match.group(1)}]")
            if len(coords) != 4:
                print("  [VISION] Element not found on screen.")
                return None
                
            ymin, xmin, ymax, xmax = coords
            
            # Normalize and convert to screen pixels
            # Gemini bounding boxes are on a 1000x1000 scale
            center_x_norm = (xmin + xmax) / 2 / 1000.0
            center_y_norm = (ymin + ymax) / 2 / 1000.0
            
            pixel_x = int(center_x_norm * self.screen_width)
            pixel_y = int(center_y_norm * self.screen_height)
            
            print(f"  [VISION] Found '{element_description}' at ({pixel_x}, {pixel_y})")
            return (pixel_x, pixel_y)

        except Exception as e:
            print(f"  [VISION] Error: {e}")
            if os.path.exists(img_path):
                os.remove(img_path)
            return None

    def click_element(self, element_description: str) -> bool:
        """Find an element visually and click it."""
        coords = self.locate_element(element_description)
        if coords:
            x, y = coords
            print(f"  [VISION] Clicking ({x}, {y})")
            # Move smoothly then click
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click()
            return True
        return False


if __name__ == "__main__":
    # Test script
    agent = VisionAgent()
    agent.click_element("The Windows Start Button")
