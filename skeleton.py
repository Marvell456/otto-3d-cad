# Webcam hand tracking used to control the live 3D model viewer.
import math
import threading
import time

import cv2

try:
    import mediapipe as mp
except ImportError:
    mp = None

CAMERA_WINDOW = "Hand Skeleton"
CAMERA_WIDTH = 420
CAMERA_HEIGHT = 300

class HandSkeletonController:
    # Track one hand in a worker thread and apply gestures on the Matplotlib thread.

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._gesture = None
        self._point = None
        self._thread = threading.Thread(target=self._camera_loop, daemon=True)

        self._last_mode = None
        self._zoom_direction = 0
        self._last_point = None
        self._last_point_time = 0.0
        self._azim = float(ax.azim)
        self._elev = float(ax.elev)

    def start(self):
        self._thread.start()
        self.fig.canvas.mpl_connect("close_event", self.stop)
        self._timer = self.fig.canvas.new_timer(interval=40)
        self._timer.add_callback(self._apply_tracking)
        self._timer.start()

    def stop(self, _event=None):
        self._stop_event.set()
        if hasattr(self, "_timer"):
            self._timer.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    @staticmethod
    def _finger_states(hand):
        return [
            hand.landmark[8].y < hand.landmark[6].y,
            hand.landmark[12].y < hand.landmark[10].y,
            hand.landmark[16].y < hand.landmark[14].y,
            hand.landmark[20].y < hand.landmark[18].y,
        ]

    def _camera_loop(self):
        if mp is None or not hasattr(mp, "solutions"):
            print("Hand tracking disabled: install mediapipe to enable skeleton.py.")
            return

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not capture.isOpened():
            print("Hand tracking disabled: could not open webcam 0.")
            return

        hands_api = mp.solutions.hands
        drawing_api = mp.solutions.drawing_utils
        with hands_api.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.6,
        ) as hands:
            cv2.namedWindow(CAMERA_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(CAMERA_WINDOW, CAMERA_WIDTH, CAMERA_HEIGHT)
            screen_width = 1920
            screen_height = 1080
            try:
                import tkinter as tk

                root = tk.Tk()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                root.destroy()
            except tk.TclError:
                pass
            cv2.moveWindow(
                CAMERA_WINDOW,
                max(screen_width - CAMERA_WIDTH - 24, 0),
                24,
            )
            try:
                while not self._stop_event.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        break
                    frame = cv2.flip(frame, 1)
                    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mode = "No hand"
                    point = None

                    if result.multi_hand_landmarks:
                        hand = result.multi_hand_landmarks[0]
                        drawing_api.draw_landmarks(
                            frame, hand, hands_api.HAND_CONNECTIONS
                        )
                        fingers = self._finger_states(hand)
                        extended = sum(fingers)
                        index_tip = hand.landmark[8]
                        point = (index_tip.x, index_tip.y)
                        if extended >= 4:
                            mode = "Open: zoom in"
                        elif extended == 0:
                            mode = "Grip: zoom out"
                        elif fingers[0] and not any(fingers[1:]):
                            mode = "Point: swipe to rotate"
                        else:
                            mode = "Tracking"

                    self._record_gesture(mode, point)

                    cv2.putText(
                        frame,
                        mode,
                        (18, 34),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (80, 235, 120),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(CAMERA_WINDOW, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop_event.set()
                        break
            finally:
                capture.release()
                cv2.destroyWindow(CAMERA_WINDOW)
                cv2.waitKey(1)

    def _record_gesture(self, mode, point):
        now = time.monotonic()
        with self._lock:
            zoom_gesture = mode in ("Open: zoom in", "Grip: zoom out")
            if zoom_gesture:
                self._zoom_direction = 1 if mode == "Open: zoom in" else -1
            else:
                self._zoom_direction = 0
            if mode != self._last_mode:
                self._last_point = None
            if mode == "Point: swipe to rotate" and point is not None:
                if self._last_point is not None:
                    elapsed = max(now - self._last_point_time, 0.001)
                    dx = point[0] - self._last_point[0]
                    dy = point[1] - self._last_point[1]
                    if abs(dx) + abs(dy) > 0.012 and elapsed < 0.25:
                        self._point = (dx, dy)
                self._last_point = point
                self._last_point_time = now
            else:
                self._last_point = None
            self._last_mode = mode

    def _apply_tracking(self):
        with self._lock:
            zoom_direction = self._zoom_direction
            movement = self._point
            self._point = None

        if zoom_direction:
            # Small per-frame changes preserve the 15% per half-second rate.
            frame_factor = 1.15 ** (0.04 / 0.5)
            self._scale_view(
                (1 / frame_factor) if zoom_direction > 0 else frame_factor
            )

        if movement is not None:
            dx, dy = movement
            self._azim -= dx * 320
            self._elev += dy * 220
            self._elev = max(-85, min(85, self._elev))
            self.ax.view_init(elev=self._elev, azim=self._azim)
            self.fig.canvas.draw_idle()

    def _scale_view(self, scale_factor):
        limits = (self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim())
        centers = [(low + high) / 2 for low, high in limits]
        spans = [(high - low) * scale_factor for low, high in limits]
        self.ax.set_xlim(centers[0] - spans[0] / 2, centers[0] + spans[0] / 2)
        self.ax.set_ylim(centers[1] - spans[1] / 2, centers[1] + spans[1] / 2)
        self.ax.set_zlim(centers[2] - spans[2] / 2, centers[2] + spans[2] / 2)
        self.fig.canvas.draw_idle()