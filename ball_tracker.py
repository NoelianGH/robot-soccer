#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Ball Tracker – Real-time HSV contour detection + Arduino robot controller  ║
║  Target : Intel NUC NUC8i3BEH  (CPU-only, no GPU, no ML model)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Pipeline:                                                                   ║
║    Webcam → GaussianBlur → HSV mask → Contour → Moments center             ║
║    → PD Controller → Serial command → Arduino IBT-2 robot                  ║
║                                                                              ║
║  Serial protocol (Arduino):                                                  ║
║    "L<pwm>\n"  – belok kiri   (pwm 0–255)                                  ║
║    "R<pwm>\n"  – belok kanan  (pwm 0–255)                                  ║
║    "F<pwm>\n"  – maju lurus   (pwm 0–255)                                  ║
║    "X\n"       – berhenti                                                   ║
║                                                                              ║
║  Hotkeys:                                                                    ║
║    O/Y/R/G/B/P/K  – ganti warna target langsung                            ║
║    M              – toggle HSV mask overlay                                 ║
║    D              – toggle debug serial log                                 ║
║    Q              – quit                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage
─────
  python ball_tracker.py                          # orange, camera 0, auto port
  python ball_tracker.py -c yellow               # warna lain
  python ball_tracker.py -p /dev/ttyUSB0         # port manual
  python ball_tracker.py -p COM3 -s 1            # Windows, kamera 1
  python ball_tracker.py --no-robot              # vision only (debug)
"""

import cv2
import numpy as np
import argparse
import sys
import time
import threading
import queue
from typing import Optional, Tuple

# ── Optional serial ───────────────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    SERIAL_OK = True
except ImportError:
    SERIAL_OK = False
    print("[WARN] pyserial not found → pip install pyserial\n"
          "       Running in vision-only mode.")


# ═════════════════════════════════════════════════════════════════════════════
# ── TUNING ───────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_WIDTH     = 640
CAM_HEIGHT    = 480
CAM_FPS       = 30

# ── Serial ────────────────────────────────────────────────────────────────────
SERIAL_BAUD   = 9600

# ── PD Controller ─────────────────────────────────────────────────────────────
#   error  = ball_cx – frame_cx   (px, positif = bola di KANAN)
#   output = Kp * error + Kd * d(error)/dt
#   output di-clamp ke [MIN_TURN … MAX_TURN]
Kp            = 0.35
Kd            = 0.08
DEAD_ZONE     = 40            # ±px dari tengah → tidak belok
MIN_TURN      = 90            # PWM minimum supaya motor tidak macet
MAX_TURN      = 210           # PWM maksimum belok

# ── Forward chase ─────────────────────────────────────────────────────────────
#   Kalau bola sudah di tengah tapi masih kecil → maju
CHASE_MIN_R   = 0.08          # radius/frame_height < ini → maju
CHASE_SPEED   = 150           # PWM maju

# ── No-ball timeout ───────────────────────────────────────────────────────────
NO_BALL_SECS  = 1.2           # detik tanpa bola → kirim STOP

# ═════════════════════════════════════════════════════════════════════════════


# ── HSV color presets ────────────────────────────────────────────────────────
#   Setiap entry: (label, hotkey, bgr_vis, [(lo_hsv, hi_hsv), ...])
#   Merah perlu dua range karena H wrap di 0/180.
COLOR_PRESETS = {
    "orange": ("ORANGE", "o", (0, 140, 255),
               [(np.array([  5, 120, 120]), np.array([ 20, 255, 255]))]),
    "yellow": ("YELLOW", "y", (0, 220, 220),
               [(np.array([ 22, 100, 100]), np.array([ 38, 255, 255]))]),
    "red":    ("RED",    "r", (0,   0, 220),
               [(np.array([  0, 120, 100]), np.array([  8, 255, 255])),
                (np.array([170, 120, 100]), np.array([180, 255, 255]))]),
    "green":  ("GREEN",  "g", (0, 210,  50),
               [(np.array([ 38,  80,  60]), np.array([ 80, 255, 255]))]),
    "blue":   ("BLUE",   "b", (220, 80,  0),
               [(np.array([ 95,  80,  40]), np.array([130, 255, 255]))]),
    "purple": ("PURPLE", "p", (200,  0, 180),
               [(np.array([125,  50,  50]), np.array([160, 255, 255]))]),
    "pink":   ("PINK",   "k", (180,  80, 220),
               [(np.array([150,  60,  60]), np.array([175, 255, 255]))]),
}

COLOR_ORDER  = ["orange", "yellow", "red", "green", "blue", "purple", "pink"]
HOTKEY_MAP   = {p[1]: k for k, p in COLOR_PRESETS.items()}   # char → key

# ── Visual constants ──────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
TRAIL_LEN   = 40
INDICATOR_H = 80
SELECTOR_H  = 34


# ═════════════════════════════════════════════════════════════════════════════
# HSV Masking
# ═════════════════════════════════════════════════════════════════════════════

def build_mask(hsv: np.ndarray, color_key: str) -> np.ndarray:
    _, _, _, ranges = COLOR_PRESETS[color_key]
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    mask = cv2.erode (mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=3)
    return mask


# ═════════════════════════════════════════════════════════════════════════════
# Ball detection  (moments-based centre, same as original algo)
# ═════════════════════════════════════════════════════════════════════════════

def detect_ball(mask: np.ndarray,
                min_radius: int) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Return (cx, cy, radius) of the largest valid contour, or (None, None, None).
    Centre = contour moments (same as original script).
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None

    c = max(cnts, key=cv2.contourArea)
    (_, _), r = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    if M["m00"] <= 0 or r < min_radius:
        return None, None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, int(r)


# ═════════════════════════════════════════════════════════════════════════════
# PD Robot Controller
# ═════════════════════════════════════════════════════════════════════════════

class RobotController:
    """
    PD controller:  output = Kp·e + Kd·(de/dt)
    Menghasilkan serial command string untuk Arduino.
    """

    def __init__(self):
        self.prev_err  = 0.0
        self.prev_time = time.time()

    def compute(self,
                cx: Optional[int],
                frame_w: int,
                frame_h: int,
                radius: Optional[int]) -> Tuple[str, str]:
        """
        Return (serial_cmd, human_label).
        serial_cmd = "X" | "L<pwm>" | "R<pwm>" | "F<pwm>"
        """
        now = time.time()
        dt  = max(now - self.prev_time, 1e-4)

        if cx is None:
            self.prev_err  = 0.0
            self.prev_time = now
            return "X", "STOP (no ball)"

        error  = cx - (frame_w // 2)       # +ve = bola di kanan
        de_dt  = (error - self.prev_err) / dt

        self.prev_err  = error
        self.prev_time = now

        if abs(error) <= DEAD_ZONE:
            # Bola sudah di tengah – cek apakah perlu maju
            if radius is not None and (radius / frame_h) < CHASE_MIN_R:
                return f"F{CHASE_SPEED}", f"FWD {CHASE_SPEED}"
            return "X", "CENTRE ✓"

        raw = Kp * error + Kd * de_dt
        spd = int(np.clip(abs(raw), MIN_TURN, MAX_TURN))

        if error > 0:
            return f"R{spd}", f"RIGHT {spd} (err={error:+.0f})"
        else:
            return f"L{spd}", f"LEFT  {spd} (err={error:+.0f})"


# ═════════════════════════════════════════════════════════════════════════════
# Serial Thread  (non-blocking, tidak menghambat vision loop)
# ═════════════════════════════════════════════════════════════════════════════

class SerialThread(threading.Thread):

    def __init__(self, port: str, baud: int = SERIAL_BAUD):
        super().__init__(daemon=True, name="SerialThread")
        self.cmd_q = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self.ser   = None
        self.port  = port
        self.baud  = baud
        self.ok    = False

    def connect(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2)           # tunggu Arduino reset
            self.ok = True
            print(f"[OK]  Serial: {self.port} @ {self.baud} baud")
            return True
        except serial.SerialException as e:
            print(f"[WARN] Serial: {e}")
            return False

    def send(self, cmd: str):
        """Non-blocking – drop kalau queue penuh."""
        try:
            self.cmd_q.put_nowait(cmd)
        except queue.Full:
            pass

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            try:
                cmd = self.cmd_q.get(timeout=0.3)
            except queue.Empty:
                continue
            if self.ser and self.ser.is_open:
                try:
                    self.ser.write((cmd + "\n").encode())
                except serial.SerialException:
                    self.ok = False


def auto_detect_port() -> Optional[str]:
    if not SERIAL_OK:
        return None
    ports = list(serial.tools.list_ports.comports())
    keywords = ("arduino", "ch340", "cp210", "ftdi", "usb serial")
    for p in ports:
        if any(k in (p.description or "").lower() for k in keywords):
            print(f"[INFO] Auto-detected: {p.device} ({p.description})")
            return p.device
    if ports:
        print(f"[INFO] No Arduino keyword, using: {ports[0].device}")
        return ports[0].device
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Draw helpers
# ═════════════════════════════════════════════════════════════════════════════

def direction_label(cx: int, frame_w: int) -> Tuple[str, tuple]:
    third = frame_w // 3
    if cx < third:
        return "◄ LEFT",  (30, 200, 255)
    elif cx > 2 * third:
        return "RIGHT ►", (30, 255, 180)
    else:
        return "CENTER",  (255, 220, 30)


def draw_trail(frame, trail: list):
    for i in range(1, len(trail)):
        t     = i / TRAIL_LEN
        thick = max(1, int(t * 5))
        col   = (int(50 + 205 * t), int(200 * t), int(255 * t))
        cv2.line(frame, trail[i - 1], trail[i], col, thick)


def draw_ball(frame, cx: int, cy: int, radius: int, frame_w: int):
    _, hue = direction_label(cx, frame_w)
    cv2.circle(frame, (cx, cy), radius + 8, hue, 2)
    cv2.circle(frame, (cx, cy), radius,     (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), 4,           hue, -1)
    cv2.putText(frame, f"({cx},{cy})", (cx + radius + 6, cy),
                FONT, 0.55, (220, 220, 220), 1)


def draw_indicator(frame, cx: Optional[int], frame_w: int):
    h, w  = frame.shape[:2]
    bar_y = h - INDICATOR_H

    ov = frame.copy()
    cv2.rectangle(ov, (0, bar_y), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    third = frame_w // 3
    for x in [third, 2 * third]:
        cv2.line(frame, (x, bar_y), (x, h), (60, 60, 80), 1)
    for lbl, rx in [("LEFT",   third // 2),
                    ("CENTER", third + third // 2),
                    ("RIGHT",  2 * third + third // 2)]:
        cv2.putText(frame, lbl, (rx - 28, bar_y + 22), FONT, 0.48, (100, 100, 130), 1)

    if cx is None:
        cv2.putText(frame, "SEARCHING...", (w // 2 - 80, bar_y + 55),
                    FONT, 0.9, (80, 80, 100), 1)
        return

    dot_x = int(np.clip(cx, 10, w - 10))
    dot_y = bar_y + INDICATOR_H // 2 + 8
    cv2.line(frame, (10, dot_y), (w - 10, dot_y), (40, 40, 60), 3)

    dir_lbl, color = direction_label(cx, frame_w)
    for r, alpha in [(22, 0.15), (14, 0.30), (8, 0.55)]:
        glow = frame.copy()
        cv2.circle(glow, (dot_x, dot_y), r, color, -1)
        cv2.addWeighted(glow, alpha, frame, 1 - alpha, 0, frame)
    cv2.circle(frame, (dot_x, dot_y), 6, (255, 255, 255), -1)
    cv2.putText(frame, dir_lbl, (w // 2 - 60, bar_y + 58), FONT, 1.1, color, 2)


def draw_color_selector(frame, active_key: str):
    """Bar warna di pojok kanan atas – hotkey + label, aktif di-highlight."""
    box_w   = 62
    pad     = 4
    start_x = frame.shape[1] - box_w * len(COLOR_ORDER) - pad

    for i, key in enumerate(COLOR_ORDER):
        label, hotkey, bgr, _ = COLOR_PRESETS[key]
        x1 = start_x + i * box_w
        x2 = x1 + box_w - 2
        y1, y2 = pad, pad + SELECTOR_H

        is_active = (key == active_key)
        bg = bgr if is_active else (25, 25, 35)
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

        border_col   = (255, 255, 255) if is_active else bgr
        border_thick = 2 if is_active else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, border_thick)

        txt     = f"[{hotkey.upper()}]{label[:3]}"
        txt_col = (10, 10, 20) if is_active else bgr
        (tw, th), _ = cv2.getTextSize(txt, FONT, 0.36, 1)
        cv2.putText(frame, txt,
                    (x1 + (box_w - 2 - tw) // 2, y1 + (SELECTOR_H + th) // 2),
                    FONT, 0.36, txt_col, 1)


def draw_hud(frame, radius: Optional[int], serial_ok: bool,
             cmd_label: str, debug_serial: bool, fps: float):
    lines = [
        f"FPS: {fps:4.1f}",
        f"TRACKING  r={radius}px" if radius else "NO BALL DETECTED",
        f"Serial: {'ON' if serial_ok else 'OFF (--no-robot)'}",
        f"CMD: {cmd_label}",
    ]
    pad    = 6
    lh     = 22
    pw     = 245
    ph     = len(lines) * lh + pad * 2
    ov     = frame.copy()
    cv2.rectangle(ov, (0, 0), (pw, ph), (15, 15, 25), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    for i, txt in enumerate(lines):
        col = (80, 255, 160)
        if i == 1:
            col = (80, 255, 160) if radius else (80, 80, 160)
        if i == 2:
            col = (80, 255, 160) if serial_ok else (80, 80, 200)
        cv2.putText(frame, txt, (pad, pad + lh * (i + 1) - 4), FONT, 0.50, col, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Main loop
# ═════════════════════════════════════════════════════════════════════════════

def run(source, color_key: str, min_radius: int, show_mask: bool,
        serial_thread: Optional["SerialThread"], no_robot: bool,
        debug_serial: bool):

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    controller  = RobotController()
    trail: list = []
    last_cmd    = ""
    cmd_label   = "INIT"
    last_seen   = 0.0

    fps_t0    = time.time()
    fps_count = 0
    fps_val   = 0.0

    print(f"\n  Ball Tracker ready  |  color={color_key}  |  min_radius={min_radius}px")
    print(f"  Robot: {'DISABLED (--no-robot)' if no_robot else 'ENABLED'}")
    print("  Hotkeys: O Y R G B P K = warna  |  M = mask  |  D = debug serial  |  Q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame   = cv2.flip(frame, 1)
        frame_w = frame.shape[1]
        frame_h = frame.shape[0]

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = build_mask(hsv, color_key)

        cx, cy, radius = detect_ball(mask, min_radius)

        # ── Trail ─────────────────────────────────────────────────────────────
        if cx is not None:
            last_seen = time.time()
            trail.append((cx, cy))
        if len(trail) > TRAIL_LEN:
            trail = trail[-TRAIL_LEN:]

        # ── PD Controller → serial ────────────────────────────────────────────
        now = time.time()
        cmd, cmd_label = controller.compute(cx, frame_w, frame_h, radius)

        # Timeout: pertahankan command terakhir sebentar setelah bola hilang
        if cx is None and (now - last_seen) < NO_BALL_SECS and last_seen > 0:
            cmd = last_cmd          # hold last command briefly

        if cmd != last_cmd:
            if serial_thread:
                serial_thread.send(cmd)
            if debug_serial:
                print(f"[SER] {cmd!r:12s}  ← {cmd_label}")
            last_cmd = cmd

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_trail(frame, trail)

        # Centre & dead-zone reference lines
        mid_x = frame_w // 2
        cv2.line(frame, (mid_x, 0), (mid_x, frame_h - INDICATOR_H), (0, 180, 255), 1)
        cv2.line(frame, (mid_x - DEAD_ZONE, 0),
                 (mid_x - DEAD_ZONE, frame_h - INDICATOR_H), (0, 100, 0), 1)
        cv2.line(frame, (mid_x + DEAD_ZONE, 0),
                 (mid_x + DEAD_ZONE, frame_h - INDICATOR_H), (0, 100, 0), 1)

        if cx is not None:
            draw_ball(frame, cx, cy, radius, frame_w)

        serial_ok = bool(serial_thread and serial_thread.ok)
        draw_hud(frame, radius, serial_ok, cmd_label, debug_serial, fps_val)
        draw_color_selector(frame, color_key)
        draw_indicator(frame, cx, frame_w)

        # Mask overlay (opsional)
        if show_mask:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame = cv2.addWeighted(frame, 0.6, mask_bgr, 0.4, 0)

        cv2.imshow("Ball Tracker", frame)

        # ── FPS ───────────────────────────────────────────────────────────────
        fps_count += 1
        if (time.time() - fps_t0) >= 1.0:
            fps_val   = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0    = time.time()

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            show_mask = not show_mask
        elif key == ord('d'):
            debug_serial = not debug_serial
            print(f"[INFO] Serial debug: {'ON' if debug_serial else 'OFF'}")
        elif chr(key) in HOTKEY_MAP:
            new_key = HOTKEY_MAP[chr(key)]
            if new_key != color_key:
                color_key = new_key
                trail.clear()
                controller.prev_err = 0.0
                print(f"[INFO] Color → {COLOR_PRESETS[color_key][0]}")

    cap.release()
    cv2.destroyAllWindows()
    print("Tracker stopped.")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ball Tracker + Arduino robot controller (NUC-optimised)")
    ap.add_argument("-s", "--source",     default=0,
                    help="Camera index (0,1,…) atau path video (default: 0)")
    ap.add_argument("-c", "--color",      default="orange",
                    choices=list(COLOR_PRESETS.keys()),
                    help="Warna bola (default: orange)")
    ap.add_argument("-r", "--min-radius", default=10, type=int,
                    help="Radius minimum bola dalam px (default: 10)")
    ap.add_argument("-p", "--port",       default=None,
                    help="Serial port Arduino (auto-detect jika kosong)")
    ap.add_argument("-b", "--baud",       default=SERIAL_BAUD, type=int,
                    help=f"Baud rate (default: {SERIAL_BAUD})")
    ap.add_argument("--no-robot",         action="store_true",
                    help="Vision only, tanpa kirim serial ke Arduino")
    ap.add_argument("-m", "--mask",       action="store_true",
                    help="Tampilkan HSV mask overlay saat startup")
    ap.add_argument("-d", "--debug",      action="store_true",
                    help="Print setiap serial command ke terminal")
    args = ap.parse_args()

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # ── Setup serial ──────────────────────────────────────────────────────────
    ser_thread: Optional[SerialThread] = None
    if not args.no_robot and SERIAL_OK:
        port = args.port or auto_detect_port()
        if port:
            ser_thread = SerialThread(port, args.baud)
            if ser_thread.connect():
                ser_thread.start()
            else:
                ser_thread = None
                print("[WARN] Serial gagal konek – lanjut vision-only.")
        else:
            print("[WARN] Tidak ada serial port ditemukan – vision-only.")

    try:
        run(
            source      = source,
            color_key   = args.color,
            min_radius  = args.min_radius,
            show_mask   = args.mask,
            serial_thread = ser_thread,
            no_robot    = args.no_robot,
            debug_serial = args.debug,
        )
    finally:
        if ser_thread:
            ser_thread.send("X")        # pastikan robot berhenti
            time.sleep(0.2)
            ser_thread.stop()