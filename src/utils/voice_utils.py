"""
voice_utils.py - Nhận diện lệnh giọng nói tiếng Việt.

Sử dụng SpeechRecognition + Google Speech API (miễn phí).
Hỗ trợ các lệnh:
- "quay trái" / "nhìn trái" / "bên trái"
- "quay phải" / "nhìn phải" / "bên phải"
- "nhìn thẳng" / "chính giữa" / "thẳng"
- "thoát" / "dừng lại" / "kết thúc"
"""

import speech_recognition as sr
import threading
import time
from typing import Optional, Callable
from enum import Enum


# ============================================================
# ĐỊNH NGHĨA LỆNH
# ============================================================

class Command(Enum):
    LOOK_LEFT = "left"
    LOOK_RIGHT = "right"
    LOOK_CENTER = "center"
    LOOK_UP = "up"
    LOOK_DOWN = "down"
    QUIT = "quit"
    UNKNOWN = "unknown"


# Từ khóa cho từng lệnh (tiếng Việt)
COMMAND_KEYWORDS = {
    Command.LOOK_LEFT: [
        "quay trái", "nhìn trái", "bên trái", "trái",
        "xoay trái", "qua trái", "sang trái", "turn left", "left",
    ],
    Command.LOOK_RIGHT: [
        "quay phải", "nhìn phải", "bên phải", "phải",
        "xoay phải", "qua phải", "sang phải", "turn right", "right",
    ],
    Command.LOOK_CENTER: [
        "nhìn thẳng", "chính giữa", "thẳng", "giữa",
        "nhìn trước", "phía trước", "center", "straight", "middle",
    ],
    Command.LOOK_UP: [
        "nhìn lên", "lên trên", "ngước lên", "up", "look up",
    ],
    Command.LOOK_DOWN: [
        "nhìn xuống", "xuống dưới", "cúi xuống", "down", "look down",
    ],
    Command.QUIT: [
        "thoát", "dừng lại", "kết thúc", "tắt", "quit", "exit", "stop",
    ],
}


def parse_command(text: str) -> Command:
    """
    Parse text giọng nói thành lệnh.

    Args:
        text: Văn bản nhận được từ STT.

    Returns:
        Command: Lệnh tương ứng.
    """
    text_lower = text.lower().strip()

    for command, keywords in COMMAND_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return command

    return Command.UNKNOWN


# ============================================================
# VOICE RECOGNIZER
# ============================================================

class VoiceRecognizer:
    """
    Lớp nhận diện giọng nói chạy nền (non-blocking).

    Cách dùng:
        rec = VoiceRecognizer()
        rec.start()
        cmd = rec.get_last_command()
        rec.stop()
    """

    def __init__(self, language: str = "vi-VN"):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.language = language

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_command: Command = Command.UNKNOWN
        self._last_text: str = ""
        self._lock = threading.Lock()

        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)

    def _listen_loop(self):
        """Vòng lặp nghe giọng nói (chạy trong thread riêng)."""
        while self._running:
            try:
                with self.microphone as source:
                    # Nghe (timeout 1s để có thể thoát)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            except sr.WaitTimeoutError:
                continue

            try:
                # Nhận diện bằng Google STT
                text = self.recognizer.recognize_google(audio, language=self.language)
                command = parse_command(text)

                with self._lock:
                    self._last_text = text
                    if command != Command.UNKNOWN:
                        self._last_command = command

            except sr.UnknownValueError:
                pass  # Không hiểu
            except sr.RequestError:
                pass  # Lỗi mạng

    def start(self):
        """Bắt đầu lắng nghe (non-blocking)."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[Voice] Dang lang nghe lenh giong noi...")

    def stop(self):
        """Dừng lắng nghe."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        print("[Voice] Da dung lang nghe.")

    def get_last_command(self) -> Command:
        """Lấy lệnh cuối cùng và reset."""
        with self._lock:
            cmd = self._last_command
            if cmd != Command.UNKNOWN:
                self._last_command = Command.UNKNOWN
            return cmd

    def get_last_text(self) -> str:
        """Lấy text cuối cùng nhận được và reset."""
        with self._lock:
            text = self._last_text
            self._last_text = ""
            return text


# ============================================================
# TEXT-TO-SPEECH (phản hồi)
# ============================================================

def speak(text: str):
    """
    Đọc text thành giọng nói (tiếng Việt).
    Sử dụng pyttsx3 (offline, miễn phí).
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(text)
        engine.runAndWait()
    except Exception:
        # Fallback: print ra console
        print(f"[TTS] {text}")
