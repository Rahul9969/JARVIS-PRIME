"""
JARVIS-PRIME Desktop Pet
=========================

A transparent, always-on-top, frameless window that hosts the 3D avatar.
Can be dragged with the mouse, and responds to programmatic commands
like jump, move, hide, etc.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import threading
import time

try:
    from PyQt6.QtCore import Qt, QUrl, QPoint, QTimer, QPropertyAnimation, QRect
    from PyQt6.QtWidgets import QApplication, QMainWindow
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtGui import QColor
except ImportError:
    print("[WARN] PyQt6 or PyQt6-WebEngine not installed. Pet will not run.")


class DesktopPet(QMainWindow):
    def __init__(self, start_pos=(100, 100)):
        super().__init__()
        
        # Make the window frameless and always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        # Make the background transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        
        # Setup WebEngine View
        self.browser = QWebEngineView(self)
        self.browser.page().setBackgroundColor(QColor(0, 0, 0, 0)) # Transparent
        
        # Load the HTML avatar
        html_path = Path(__file__).parent.parent / "assistant" / "avatar" / "index.html"
        self.browser.setUrl(QUrl(f"file:///{html_path.absolute().as_posix()}"))
        
        self.setCentralWidget(self.browser)
        self.resize(380, 420)
        self.move(*start_pos)
        
        # Variables for dragging
        self._drag_active = False
        self._drag_pos = QPoint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_active:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_active = False
        event.accept()

    # --- Programmable Actions ---

    def jump(self):
        """Make the pet jump."""
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(500)
        current = self.geometry()
        
        # Go up then down
        self.anim.setKeyValueAt(0.0, current)
        self.anim.setKeyValueAt(0.5, QRect(current.x(), current.y() - 150, current.width(), current.height()))
        self.anim.setKeyValueAt(1.0, current)
        self.anim.start()

    def move_to(self, x, y, duration=1000):
        """Smoothly move to a new location."""
        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(duration)
        current = self.geometry()
        self.anim.setStartValue(current)
        self.anim.setEndValue(QRect(x, y, current.width(), current.height()))
        self.anim.start()

    def set_state(self, state: str):
        """Change avatar state (listening, speaking, thinking, etc)."""
        self.browser.page().runJavaScript(f"setState('{state}');")
        
    def show_message(self, text: str, duration: int = 5000):
        """Show a chat bubble."""
        # Escape quotes
        safe_text = text.replace("'", "\\'")
        self.browser.page().runJavaScript(f"showBubble('{safe_text}', {duration});")


class PetController:
    """Controls the desktop pet in a separate thread so it doesn't block."""
    def __init__(self):
        self.app = None
        self.pet = None
        self._thread = None
        self._ready = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run_app, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_app(self):
        # Must run in its own thread if main thread is blocked, 
        # but Qt usually requires main thread. 
        # For this architecture, we run Qt in this thread and handle events.
        self.app = QApplication(sys.argv)
        self.pet = DesktopPet(start_pos=(1500, 600))
        self.pet.show()
        self._ready.set()
        self.app.exec()

    def jump(self):
        if self.pet:
            # Must use invokeMethod to interact across threads safely in real apps,
            # but for simple calls QTimer.singleShot works
            QTimer.singleShot(0, self.pet.jump)

    def set_state(self, state):
        if self.pet:
            QTimer.singleShot(0, lambda: self.pet.set_state(state))
            
    def speak(self, text):
        if self.pet:
            QTimer.singleShot(0, lambda: self.pet.show_message(text))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    
    # Test jump
    QTimer.singleShot(2000, pet.jump)
    QTimer.singleShot(4000, lambda: pet.show_message("I am alive!"))
    
    sys.exit(app.exec())
