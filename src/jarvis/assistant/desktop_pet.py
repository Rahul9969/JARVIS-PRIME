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
    from PyQt6.QtCore import Qt, QUrl, QPoint, QTimer, QPropertyAnimation, QRect, pyqtSignal
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLineEdit
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtGui import QColor
except ImportError:
    print("[WARN] PyQt6 or PyQt6-WebEngine not installed. Pet will not run.")


class TransparentOverlay(QWidget):
    text_command_submitted = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.window = parent
        self._drag_active = False
        self._drag_pos = QPoint()
        self._has_dragged = False
        
        self.input_field = QLineEdit(self)
        self.input_field.setPlaceholderText("Type a command...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(20, 20, 20, 220);
                color: #00ffcc;
                border: 2px solid #00ffcc;
                border-radius: 12px;
                padding: 6px;
                font-family: Consolas, monospace;
                font-size: 14px;
            }
        """)
        self.input_field.hide()
        self.input_field.returnPressed.connect(self._on_submit)
        
    def _on_submit(self):
        text = self.input_field.text().strip()
        if text:
            self.text_command_submitted.emit(text)
        self.input_field.clear()
        self.input_field.hide()

    def resizeEvent(self, event):
        self.input_field.setGeometry(20, self.height() - 80, self.width() - 40, 36)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._has_dragged = False
            self._drag_pos = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_active:
            self._has_dragged = True
            self.window.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = False
            if not self._has_dragged:
                if self.input_field.isHidden():
                    self.input_field.show()
                    self.input_field.setFocus()
                else:
                    self.input_field.hide()
            event.accept()
        
    def wheelEvent(self, event):
        self.window.wheelEvent(event)


class DesktopPet(QMainWindow):
    state_changed = pyqtSignal(str)
    message_requested = pyqtSignal(str)
    jump_requested = pyqtSignal()
    move_left_requested = pyqtSignal()
    move_right_requested = pyqtSignal()
    scale_requested = pyqtSignal(float)
    expression_requested = pyqtSignal(str)
    hide_requested = pyqtSignal()
    show_requested = pyqtSignal()
    shake_requested = pyqtSignal()
    text_command_requested = pyqtSignal(str)

    def __init__(self, start_pos=(100, 100)):
        super().__init__()
        
        # Connect thread-safe signals
        self.state_changed.connect(self.set_state)
        self.message_requested.connect(self.show_message)
        self.jump_requested.connect(self.jump)
        self.move_left_requested.connect(lambda: self.move_to(max(0, self.geometry().x() - 200), self.geometry().y()))
        self.move_right_requested.connect(lambda: self.move_to(min(3000, self.geometry().x() + 200), self.geometry().y()))
        self.scale_requested.connect(self.scale_pet)
        self.expression_requested.connect(self.set_expression)
        self.hide_requested.connect(self.hide_pet)
        self.show_requested.connect(self.show_pet)
        self.shake_requested.connect(self.shake_pet)
        
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
        
        self.overlay = TransparentOverlay(self)
        self.overlay.resize(self.size())
        self.overlay.text_command_submitted.connect(self.text_command_requested.emit)
        self.overlay.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.resize(event.size())

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # Scale up or down by 10%
        factor = 1.1 if delta > 0 else 0.9
        
        new_width = int(self.width() * factor)
        new_height = int(self.height() * factor)
        
        if 150 < new_width < 1000:
            self.resize(new_width, new_height)
            zoom = self.browser.zoomFactor()
            self.browser.setZoomFactor(zoom * factor)
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

    def hide_pet(self):
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.start()

    def show_pet(self):
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.start()

    def shake_pet(self):
        self.anim = QPropertyAnimation(self, b"pos")
        self.anim.setDuration(300)
        current = self.pos()
        self.anim.setKeyValueAt(0.0, current)
        self.anim.setKeyValueAt(0.25, QPoint(current.x() - 20, current.y()))
        self.anim.setKeyValueAt(0.5, QPoint(current.x() + 20, current.y()))
        self.anim.setKeyValueAt(0.75, QPoint(current.x() - 20, current.y()))
        self.anim.setKeyValueAt(1.0, current)
        self.anim.start()

    def scale_pet(self, factor):
        """Scale programmatically."""
        new_width = int(self.width() * factor)
        new_height = int(self.height() * factor)
        if 150 < new_width < 1000:
            self.resize(new_width, new_height)
            zoom = self.browser.zoomFactor()
            self.browser.setZoomFactor(zoom * factor)

    def set_state(self, state: str):
        """Change avatar state (listening, speaking, thinking, etc)."""
        self.browser.page().runJavaScript(f"setState('{state}');")
        
    def set_expression(self, expr: str):
        """Change facial expression."""
        self.browser.page().runJavaScript(f"setExpression('{expr}');")
        
    def show_message(self, text: str, duration: int = 5000):
        """Show a chat bubble."""
        # Escape quotes and newlines to prevent JS SyntaxError
        safe_text = text.replace("'", "\\'").replace("\n", " ")
        self.browser.page().runJavaScript(f"showBubble('{safe_text}', {duration});")


class PetController:
    """Controls the desktop pet in a separate thread so it doesn't block."""
    def __init__(self):
        self.app = None
        self.pet = None
        self._thread = None
        self._ready = threading.Event()
        self.text_command_callback = None

    def start(self):
        self._thread = threading.Thread(target=self._run_app, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def start_sync(self):
        self.app = QApplication(sys.argv)
        self.pet = DesktopPet(start_pos=(100, 100))
        self.pet.text_command_requested.connect(self._handle_text_command)
        self.pet.show()
        return self.app.exec()

    def _run_app(self):
        self.app = QApplication(sys.argv)
        self.pet = DesktopPet(start_pos=(100, 100))
        self.pet.text_command_requested.connect(self._handle_text_command)
        self.pet.show()
        self._ready.set()
        self.app.exec()

    def _handle_text_command(self, text):
        if self.text_command_callback:
            self.text_command_callback(text)

    def jump(self):
        if self.pet:
            self.pet.jump_requested.emit()

    def move_left(self):
        if self.pet:
            self.pet.move_left_requested.emit()

    def move_right(self):
        if self.pet:
            self.pet.move_right_requested.emit()

    def hide(self):
        if self.pet:
            self.pet.hide_requested.emit()

    def show(self):
        if self.pet:
            self.pet.show_requested.emit()

    def shake(self):
        if self.pet:
            self.pet.shake_requested.emit()

    def scale(self, factor):
        if self.pet:
            self.pet.scale_requested.emit(factor)

    def set_state(self, state):
        if self.pet:
            self.pet.state_changed.emit(state)
            
    def set_expression(self, expr):
        if self.pet:
            self.pet.expression_requested.emit(expr)

    def speak(self, text):
        if self.pet:
            self.pet.message_requested.emit(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    
    # Test jump
    QTimer.singleShot(2000, pet.jump)
    QTimer.singleShot(4000, lambda: pet.show_message("I am alive!"))
    
    sys.exit(app.exec())
