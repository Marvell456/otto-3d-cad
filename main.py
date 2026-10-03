# Push-to-talk voice assistant with AI-driven intent routing.
import json
import os
import re
import subprocess
import sys
import threading
import tkinter as tk
from dotenv import load_dotenv
from openai import OpenAI
import pyaudio
import pyttsx3
import speech_recognition as sr

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

def _make_ollama_client():
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )

ollama_client = _make_ollama_client()

def ollama_completion(**kwargs):
    return ollama_client.chat.completions.create(**kwargs)

# Prompt for zero-shot intent classification.
ROUTER_SYSTEM_PROMPT = """
You are an intent classification engine for an AI assistant. Analyze the user's voice prompt and classify it into exactly ONE of the following three actions.

ACTIONS:
1. "CAMERA": User wants to open, start, launch, or use the webcam/camera to inspect an object and create a 3D model from what it sees.
2. "GENERATE_3D": User wants to create, visualize, render, model, draw, make, show, or generate any physical object/shape in 3D (e.g., house, car, plane, bottle, pencil, table, cup, chair, sword, etc.).
3. "CHAT": General conversation, factual questions, greetings, coding help, or anything else not involving the camera or building a 3D model.

OUTPUT FORMAT:
Return ONLY a JSON object with two keys:
- "action": string ("CAMERA", "GENERATE_3D", or "CHAT")
- "clean_prompt": string (If GENERATE_3D, extract ONLY the target object name like 'pencil', 'red sports car', 'house'. Otherwise set to null.)
"""

class PushToTalkApp:

    @staticmethod
    def _is_quota_error(error):
        return (
            getattr(error, "status_code", None) == 429
            or "quota" in str(error).lower()
            or "rate limit" in str(error).lower()
        )

    @staticmethod
    def _quota_message(error=None):
        if error is not None:
            retry_match = re.search(
                r"retry(?: again)? in\s+([0-9]+(?:\.[0-9]+)?)s",
                str(error),
                flags=re.IGNORECASE,
            )
            if retry_match:
                seconds = max(1, round(float(retry_match.group(1))))
                return (
                    "Ollama request limit reached. "
                    f"Please wait about {seconds} seconds, then try again."
                )
        return (
            "Ollama request limit reached. Please wait, then try again."
        )

    def __init__(self, root):
        self.root = root
        self.root.title("OTTO • Voice Control")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#0B0E14")

        # Audio setup
        self.recognizer = sr.Recognizer()
        self.pa = pyaudio.PyAudio()
        self.rate = 16000
        self.chunk = 1024
        self.channels = 1
        self.audio_format = pyaudio.paInt16

        # State
        self.is_recording = False
        self.key_held = False
        self.pending_release_id = None
        self.frames = []
        self.audio_thread = None
        self.quota_retry_message = None

        self._build_ui()

        # 'q' key bindings
        self.root.bind("<KeyPress-q>", self.on_key_press)
        self.root.bind("<KeyRelease-q>", self.on_key_release)
        self.root.focus_set()

    def _build_ui(self):
        # Compact dark command-center interface.
        self.root.option_add("*Font", ("Segoe UI", 10))
        header = tk.Frame(self.root, bg="#111827", height=68)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        title_lbl = tk.Label(
            header,
            text="OTTO",
            font=("Segoe UI", 18, "bold"),
            fg="#00F0FF",
            bg="#111827",
        )
        title_lbl.pack(side="left", padx=(24, 8), pady=14)

        subtitle_lbl = tk.Label(
            header,
            text="VOICE COMMAND CENTER  •  OLLAMA",
            font=("Segoe UI", 9, "bold"),
            fg="#64748B",
            bg="#111827",
        )
        subtitle_lbl.pack(side="left", pady=18)

        self.status_lbl = tk.Label(
            header,
            text="● STANDBY",
            font=("Segoe UI", 10, "bold"),
            fg="#6B7280",
            bg="#111827",
        )
        self.status_lbl.pack(side="right", padx=24, pady=22)

        container = tk.Frame(self.root, bg="#0B0E14")
        container.pack(expand=True, fill="both", padx=20, pady=(20, 12))
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        # Left panel: user speech text
        left_panel = tk.Frame(
            container,
            bg="#131924",
            highlightbackground="#1F293D",
            highlightthickness=1,
        )
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        user_title = tk.Label(
            left_panel,
            text="USER VOICE INPUT",
            font=("Consolas", 9, "bold"),
            fg="#00F0FF",
            bg="#131924",
        )
        user_title.pack(anchor="nw", padx=18, pady=(16, 8))

        self.user_box = tk.Text(
            left_panel,
            bg="#131924",
            fg="#E5E7EB",
            font=("Segoe UI", 12),
            bd=0,
            relief="flat",
            wrap="word",
            padx=18,
            pady=12,
            insertbackground="#00F0FF",
            selectbackground="#164E63",
        )
        self.user_box.pack(expand=True, fill="both", padx=8, pady=(0, 8))
        self.user_box.insert(
            "1.0", "Hold 'Q' and speak... transcribed text will appear here."
        )
        self.user_box.config(state="disabled")

        # Right panel: system actions & voice responses
        right_panel = tk.Frame(
            container,
            bg="#131924",
            highlightbackground="#1F293D",
            highlightthickness=1,
        )
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        ai_title = tk.Label(
            right_panel,
            text="SYSTEM RESPONSE",
            font=("Consolas", 9, "bold"),
            fg="#7C3AED",
            bg="#131924",
        )
        ai_title.grid(
            row=0, column=0, sticky="w", padx=18, pady=(16, 8)
        )

        self.ai_box = tk.Text(
            right_panel,
            bg="#131924",
            fg="#A7F3D0",
            font=("Segoe UI", 12),
            bd=0,
            relief="flat",
            wrap="word",
            padx=18,
            pady=12,
            insertbackground="#A7F3D0",
            selectbackground="#14532D",
        )
        self.ai_box.grid(
            row=1, column=0, sticky="nsew", padx=8, pady=(0, 8)
        )
        self.ai_box.insert(
            "1.0", "System status, actions, and spoken replies appear here."
        )
        self.ai_box.config(state="disabled")

    # Key handling
    def on_key_press(self, event):
        if self.pending_release_id is not None:
            self.root.after_cancel(self.pending_release_id)
            self.pending_release_id = None

        if not self.key_held:
            self.key_held = True
            self.start_recording()

    def on_key_release(self, event):
        # 0.5s delay to ensure trailing spoken words are captured.
        if self.pending_release_id is not None:
            self.root.after_cancel(self.pending_release_id)
        self.pending_release_id = self.root.after(500, self._confirm_release)

    def _confirm_release(self):
        self.pending_release_id = None
        self.key_held = False
        self.stop_recording()

    # Recording
    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.status_lbl.config(text="● LISTENING...", fg="#EF4444")
        self.user_box.delete("1.0", tk.END)
        self.user_box.insert("1.0", "Listening...")
        self.audio_thread = threading.Thread(
            target=self._record_loop, daemon=True
        )
        self.audio_thread.start()

    def _record_loop(self):
        stream = self.pa.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )
        while self.is_recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except Exception:
                break
        stream.stop_stream()
        stream.close()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=2)
        self.process_recording()

    # Dynamic AI intent classification
    @staticmethod
    def _is_math_request(user_text):
        # Keep math questions on the normal chat path.
        text = user_text.lower()
        math_terms = (
            "calculate",
            "calculation",
            "equation",
            "solve",
            "math",
            "mathematics",
            "algebra",
            "multiply",
            "times",
            "plus",
            "minus",
            "subtract",
            "divide",
            "percentage",
            "percent",
            "square root",
        )
        if any(term in text for term in math_terms):
            return True
        return bool(
            re.search(r"\d\s*[\+\-\*\/\^=]\s*\d", text)
            or re.search(r"\bwhat is\s+\d", text)
        )

    def _classify_intent(self, user_text):
        # Classify commands as 3D generation or ordinary conversation.
        if re.search(
            r"\b(open|start|launch|use|show|turn on)\b.{0,24}\b(camera|webcam)\b"
            r"|\b(camera|webcam)\b.{0,24}\b(open|start|launch|use|show|turn on)\b",
            user_text,
            flags=re.IGNORECASE,
        ):
            return "CAMERA", None
        if self._is_math_request(user_text):
            return "CHAT", None
        try:
            response = ollama_completion(
                model=OLLAMA_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("action", "CHAT"), data.get("clean_prompt")
        except Exception as e:
            print(f"Router Exception: {e}")
            if self._is_quota_error(e):
                self.quota_retry_message = self._quota_message(e)
                return "QUOTA_LIMIT", None
            return "CHAT", None

    def process_recording(self):
        if not self.frames:
            self.status_lbl.config(text="● STANDBY", fg="#6B7280")
            return

        self.status_lbl.config(text="● TRANSCRIBING...", fg="#F59E0B")
        raw_audio = b"".join(self.frames)
        audio_data = sr.AudioData(
            raw_audio, self.rate, self.pa.get_sample_size(self.audio_format)
        )
        threading.Thread(
            target=self._process_speech, args=(audio_data,), daemon=True
        ).start()

    def _process_speech(self, audio_data):
        try:
            user_text = self.recognizer.recognize_google(audio_data).strip()
        except sr.UnknownValueError:
            user_text = "(Could not understand audio)"
        except sr.RequestError as e:
            user_text = f"(Speech service error: {e})"

        self.root.after(0, lambda: self._update_user_box(user_text))

        if user_text.startswith("("):
            self.root.after(
                0,
                lambda: self.status_lbl.config(text="● STANDBY", fg="#6B7280"),
            )
            return

        # Let the local Ollama model determine the action dynamically.
        self.root.after(
            0,
            lambda: self.status_lbl.config(
                text="● ROUTING...", fg="#8B5CF6"
            ),
        )
        action, extracted_target = self._classify_intent(user_text)

        if action == "QUOTA_LIMIT":
            quota_message = self.quota_retry_message or self._quota_message()
            self.quota_retry_message = None
            self.root.after(0, lambda: self._update_ai_box(quota_message))
            self._speak_text(quota_message)
            self.root.after(
                0,
                lambda: self.status_lbl.config(text="● QUOTA LIMIT", fg="#F59E0B"),
            )
            return

        # Route 1: Open camera-to-3D capture tool.
        if action == "CAMERA":
            response_msg = "Opening camera-to-3D tool..."
            self.root.after(0, lambda: self._update_ai_box(response_msg))
            self._speak_text(response_msg)
            subprocess.Popen([sys.executable, "camera.py"])

        # Route 2: Generate 3D shape.
        elif action == "GENERATE_3D":
            target_object = extracted_target if extracted_target else user_text
            voice_alert = (
                f"Please wait, the 3D model of {target_object} is being generated..."
            )
            display_text = f"{voice_alert}\n\nObject: '{target_object}'"

            self.root.after(0, lambda: self._update_ai_box(display_text))
            self._speak_text(voice_alert)

            self.root.after(
                0,
                lambda: self._launch_shape_generation(target_object),
            )

        # Route 3: General AI conversation via Ollama.
        else:
            self.root.after(
                0,
                lambda: self.status_lbl.config(
                    text="● THINKING...", fg="#8B5CF6"
                ),
            )
            try:
                response = ollama_completion(
                    model=OLLAMA_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a concise voice assistant. "
                                "Keep your answers brief, clear, and direct. "
                                "Respond in 1 to 3 short sentences maximum."
                            ),
                        },
                        {"role": "user", "content": user_text},
                    ],
                )
                ai_reply = response.choices[0].message.content
            except Exception as e:
                ai_reply = (
                    self._quota_message(e)
                    if self._is_quota_error(e)
                    else f"Error calling Ollama ({OLLAMA_MODEL}): {e}"
                )

            self.root.after(0, lambda: self._update_ai_box(ai_reply))
            self._speak_text(ai_reply)

        self.root.after(
            0,
            lambda: self.status_lbl.config(text="● STANDBY", fg="#6B7280"),
        )

    def _speak_text(self, text):
        # Run TTS in a worker thread cleanly.
        self.root.after(
            0,
            lambda: self.status_lbl.config(text="● SPEAKING...", fg="#10B981"),
        )
        try:
            tts = pyttsx3.init()
            tts.setProperty("rate", 175)
            tts.say(text)
            tts.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

    def _update_user_box(self, text):
        self.user_box.config(state="normal")
        self.user_box.delete("1.0", tk.END)
        self.user_box.insert("1.0", text)
        self.user_box.config(state="disabled")

    def _update_ai_box(self, text):
        self.ai_box.config(state="normal")
        self.ai_box.delete("1.0", tk.END)
        self.ai_box.insert("1.0", text)
        self.ai_box.config(state="disabled")

    def _launch_shape_generation(self, target_object):
        subprocess.Popen([sys.executable, "shapes.py", target_object])

    def on_close(self):
        self.is_recording = False
        try:
            self.pa.terminate()
        except Exception:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PushToTalkApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()