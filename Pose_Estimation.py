"""
Usage:
  python Pose_Estimation.py --model movenet_lightning.tflite --debug
  python Pose_Estimation.py --skip-frames 2  # run model every 2nd frame; can do more at the cost of lag

Tuning:
  Threshold tuning example to increase sensitivity: python Pose_Estimation.py --torso-thresh 45 --ratio-thresh 0.9 --hf-thresh 50
  Angled camera (e.g. 15deg clockwise tilt): python Pose_Estimation.py --camera-tilt 15
  High camera (e.g. mounted at 30deg above horizontal): python Pose_Estimation.py --camera-elevation 30

"""

import time
import socket as _socket
import os
import websocket
import json
import argparse
import io
import csv
import urllib.request
import urllib.error
from collections import deque
import math
import threading
import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

# Constants
SKELETON_CONNECTIONS = np.array([
    [0,5],[0,6],[5,6],[5,7],[7,9],[6,8],[8,10],
    [5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],
], dtype=np.int32)

SKELETON_COLOR = (0, 255, 0)

# Keypoint indexes 
HEAD_IDX = np.array([0,1,2,3,4])
FEET_IDX = np.array([15,16])
BODY_IDX = np.array([0,5,6,11,12,13,14,15,16])
LS, RS, LH, RH = 5, 6, 11, 12

# Upper body: head keypoints + shoulders + elbows + wrists
# Lower body: hips + knees + ankles
UPPER_BODY_IDX = np.array([0,1,2,3,4, 5,6, 7,8, 9,10])
LOWER_BODY_IDX = np.array([11,12, 13,14, 15,16])

# WebSocket state (shared between ws thread and main loop)
class SocketState:
    HINT_TTL = 15.0  # seconds a fall hint from the phone stays active

    CSV_PREVIEW_LINES = 4  # CSV rows to show in debug panel

    def __init__(self):
        self._lock = threading.Lock()
        self.connected = False
        self.last_result = None    # 'fall', 'no_fall', 'error'
        self.last_msg_t = None     # timestamp of last result
        self._ws = None
        self.csv_preview = []      # last CSV rows 
        self.csv_url = None        # last CSV URL received

    def set_ws(self, ws):
        with self._lock:
            self._ws = ws

    def set_connected(self, val):
        with self._lock:
            self.connected = val

    def set_csv_preview(self, url, rows):
        with self._lock:
            self.csv_url = url
            self.csv_preview = rows[:self.CSV_PREVIEW_LINES]

    def set_result(self, result):
        with self._lock:
            self.last_result = result
            self.last_msg_t = time.time()

    def send(self, msg):
        with self._lock:
            ws = self._ws
        if ws:
            try:
                ws.send(msg)
            except Exception:
                pass

    def snapshot(self):
        with self._lock:
            now = time.time()
            hint = (
                self.last_result == 'fall'
                and self.last_msg_t is not None
                and (now - self.last_msg_t) < self.HINT_TTL
            )
            return {
                'connected': self.connected,
                'last_result': self.last_result,
                'last_msg_t': self.last_msg_t,
                'hint': hint,
                'csv_url': self.csv_url,
                'csv_preview': list(self.csv_preview),
            }


# IMU-based fall analysis
class IMUAnalyzer:
    """Analyze accelerometer/gyroscope CSV data for fall indicators.

    Computes Sum Vector Magnitude (SVM) from raw accelerometer readings
    and finds free-fall + impact patterns.

    Tilt is measured relative to a learned standing baseline so the sensor
    orientation does not need to be known in advance.  The first few packets
    where SVM ~ 1g (person is upright and still) are used to record the
    gravity vector direction; subsequent packets measure how far the current
    gravity vector has rotated from that reference.
    """
    ACCEL_SCALE = 16384.0   # LSB/g for +/-2g range
    GYRO_SCALE = 131.0      # LSB/(deg/s) for +/-250deg/s range

    # Thresholds (in g and deg/s)
    FREEFALL_THRESH = 0.5   # SVM below this = free fall
    IMPACT_THRESH = 2.0     # SVM above this = impact
    GYRO_THRESH = 150.0     # angular velocity spike
    SVM_STD_THRESH = 0.3    # high variance = turbulent motion
    TILT_THRESH = 60.0      # degrees from standing baseline -> phone horizontal
    GYRO_ROTATION_THRESH = 45.0  # cumulative degrees in window -> rapid tumble

    # Baseline calibration
    BASELINE_PACKETS = 5    # number of quiet upright packets to average for baseline

    BASELINE_FILE = '/tmp/imu_baseline.npy'

    def __init__(self):
        self.last_score = 0.0
        self.last_score_t = 0.0
        self.last_tilt = 0.0
        self._baseline_gravity = None
        self._baseline_samples = []

        # Load saved baseline so calibration survives restarts
        if os.path.exists(self.BASELINE_FILE):
            try:
                self._baseline_gravity = np.load(self.BASELINE_FILE)
                print(f"[IMU] Baseline loaded from disk: {self._baseline_gravity.round(3)}")
            except Exception:
                pass

        if self._baseline_gravity is None:
            print("[IMU] No baseline found - stand still for ~25s to calibrate")

    def analyze(self, csv_content):
        """Analyze IMU CSV and return a fall score 0.0-1.0.

        Returns:
            float: 0.0 = no fall indicators, 1.0 = strong fall indicators
        """
        rows = []
        reader = csv.reader(io.StringIO(csv_content))
        for row in reader:
            if len(row) < 6:
                continue
            try:
                vals = [float(v) for v in row[:6]]
                rows.append(vals)
            except ValueError:
                continue  # skip header

        if len(rows) < 5:
            return 0.0

        data = np.array(rows)
        ax = data[:, 0] / self.ACCEL_SCALE
        ay = data[:, 1] / self.ACCEL_SCALE
        az = data[:, 2] / self.ACCEL_SCALE
        gx = data[:, 3] / self.GYRO_SCALE
        gy = data[:, 4] / self.GYRO_SCALE
        gz = data[:, 5] / self.GYRO_SCALE

        svm = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

        score = 0.0

        # Free fall detection--> SVM drops near 0g
        freefall_count = np.sum(svm < self.FREEFALL_THRESH)
        if freefall_count >= 2:
            score += 0.35

        # Impact spike--> SVM exceeds threshold
        impact_peak = svm.max()
        if impact_peak > self.IMPACT_THRESH:
            # Scale contribution--> 2g = 0.2, 4g+ = 0.35
            score += min(0.35, 0.2 + (impact_peak - self.IMPACT_THRESH) * 0.075)

        # Gyroscope spike: rapid rotation during fall event
        gyro_peak = gyro_mag.max()
        if gyro_peak > self.GYRO_THRESH:
            score += min(0.2, (gyro_peak - self.GYRO_THRESH) / 500.0 * 0.2)

        # Gyroscope cumulative rotation: integrate angular velocity over window
        # Large cumulative rotation = person tumbled/rotated rapidly
        n = len(gx)
        dt = 0.2 / max(n, 1)  # assume 0.2s window, uniform sampling
        cum_rot = max(
            abs(np.cumsum(gx * dt)).max(),
            abs(np.cumsum(gy * dt)).max(),
            abs(np.cumsum(gz * dt)).max(),
        )
        if cum_rot > self.GYRO_ROTATION_THRESH:
            score += min(0.25, (cum_rot - self.GYRO_ROTATION_THRESH) / 90.0 * 0.25)

        # Tilt detection relative to learned standing baseline
        mean_accel = np.array([ax.mean(), ay.mean(), az.mean()])
        mean_svm = float(np.linalg.norm(mean_accel))

        if self._baseline_gravity is None:
            # Calibration phase: collect quiet upright packets (SVM ~ 1g, low variance)
            if 0.85 < mean_svm < 1.15 and svm.std() < 0.05:
                self._baseline_samples.append(mean_accel / (mean_svm + 1e-6))
                if len(self._baseline_samples) >= self.BASELINE_PACKETS:
                    self._baseline_gravity = np.mean(self._baseline_samples, axis=0)
                    self._baseline_gravity /= np.linalg.norm(self._baseline_gravity)
                    print(f"[IMU] Standing baseline calibrated: {self._baseline_gravity.round(3)}")
                    try:
                        np.save(self.BASELINE_FILE, self._baseline_gravity)
                        print(f"[IMU] Baseline saved to {self.BASELINE_FILE}")
                    except Exception as e:
                        print(f"[IMU] Could not save baseline: {e}")
            mean_tilt = 0.0  # no tilt score until calibrated
        else:
            # Measure angle between current gravity vector and standing baseline
            cur_gravity = mean_accel / (mean_svm + 1e-6)
            dot = float(np.clip(np.dot(cur_gravity, self._baseline_gravity), -1.0, 1.0))
            mean_tilt = math.degrees(math.acos(dot))

        if mean_tilt > self.TILT_THRESH:
            score += min(0.35, (mean_tilt - self.TILT_THRESH) / 30.0 * 0.35)

        # SVM variance: high variance = turbulent motion (fall + impact)
        svm_std = svm.std()
        if svm_std > self.SVM_STD_THRESH:
            score += 0.1

        self.last_tilt = mean_tilt
        self.last_score = min(1.0, score)
        self.last_score_t = time.time()
        return self.last_score

    def get_hint(self, ttl=15.0):
        """Return current IMU score if recent else 0."""
        if (time.time() - self.last_score_t) < ttl:
            return self.last_score
        return 0.0


# IPC server for peripheral controller
class IPCServer:
    """Unix domain socket server for peripheral controller communication.
    """

    def __init__(self, sock_path='/tmp/falldetect.sock'):
        self.sock_path = sock_path
        self._lock = threading.Lock()
        self._client = None
        self._server = None
        self._running = False
        self._on_command = lambda msg: None

    def start(self):
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._server = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self._server.bind(self.sock_path)
        self._server.listen(1)
        self._server.settimeout(1.0)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while self._running:
            try:
                client, _ = self._server.accept()
            except OSError:
                continue
            with self._lock:
                if self._client:
                    try:
                        self._client.close()
                    except Exception:
                        pass
                self._client = client
            print("[IPC] Peripheral controller connected")
            threading.Thread(target=self._recv_loop, args=(client,), daemon=True).start()

    def _recv_loop(self, client):
        buf = b''
        while self._running:
            try:
                data = client.recv(4096)
                if not data:
                    break
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    try:
                        self._on_command(json.loads(line.decode()))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except Exception:
                break
        with self._lock:
            if self._client is client:
                self._client = None
        print("[IPC] Peripheral controller disconnected")

    def set_command_callback(self, cb):
        self._on_command = cb

    def send(self, msg_dict):
        with self._lock:
            client = self._client
        if client:
            try:
                client.sendall((json.dumps(msg_dict) + '\n').encode())
            except Exception:
                with self._lock:
                    self._client = None

    def stop(self):
        self._running = False
        try:
            self._server.close()
        except Exception:
            pass
        try:
            os.unlink(self.sock_path)
        except Exception:
            pass


class SnapshotServer:
    """Upload fall snapshot JPEGs to the ngrok server and return the public URL.

    POSTs a JPEG to  https://<base_url>/upload  (follows any redirect the server
    returns) and returns  https://<base_url>/uploads/fall_<ts>.jpg  for the phone.
    """

    def __init__(self, base_url='1327-68-148-232-205.ngrok-free.app'):
        self.base_url = base_url.rstrip('/').removeprefix('https://').removeprefix('http://')

    def start(self):
        print(f"[SNAP] Snapshot upload -> https://{self.base_url}")

    def capture(self, frame):
        """Encode frame as JPEG, upload to ngrok server, return public URL."""
        ts = int(time.time())
        filename = f"fall_{ts}.jpg"

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print("[SNAP] Failed to encode frame")
            return None

        jpg_bytes = buf.tobytes()
        boundary = "FallDetectBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + jpg_bytes + f"\r\n--{boundary}--\r\n".encode()

        # Server upload endpoint is POST / (root), file served at /uploads/<name>
        protocol = 'http' if self.base_url.startswith('localhost') or self.base_url.startswith('127.') else 'https'
        req = urllib.request.Request(
            f"{protocol}://{self.base_url}/",
            data=body,
            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:
            print(f"[SNAP] Upload failed: {e}")
            return None

        public_url = f"https://{self.base_url}/uploads/{filename}"
        print(f"[SNAP] Uploaded (HTTP {status}) -> {public_url}")
        return public_url


# Camera tilt auto-calibrator
class CameraCalibrator:
    """Auto-calibrate camera roll (tilt) from standing-pose skeleton geometry.

    When a person stands upright the shoulder->hip vector should point straight down.
    Any measured deviation from vertical equals the camera roll angle.

    Samples are collected whenever the person appears roughly upright
    (torso clearly more vertical than horizontal).  After MIN_SAMPLES frames
    the median angle is committed as the tilt correction and re-averaged every
    RECALIBRATE_INTERVAL additional upright frames.
    """

    MIN_SAMPLES = 30
    RECALIBRATE_INTERVAL = 300  # re-average every 300 upright frames (~10 s at 30 fps)

    def __init__(self):
        self._samples = deque(maxlen=120)
        self.tilt_deg = 0.0
        self.calibrated = False
        self._frame_count = 0

    def feed(self, raw_kps, thresh=0.4):
        """Feed raw (uncorrected) keypoints from a frame that looks roughly upright."""
        confs = raw_kps[:, 2]
        has_ls = confs[LS] >= thresh
        has_rs = confs[RS] >= thresh
        has_lh = confs[LH] >= thresh
        has_rh = confs[RH] >= thresh

        if not (has_ls or has_rs) or not (has_lh or has_rh):
            return

        if has_ls and has_rs:
            sh = (raw_kps[LS, :2] + raw_kps[RS, :2]) / 2
        elif has_ls:
            sh = raw_kps[LS, :2].copy()
        else:
            sh = raw_kps[RS, :2].copy()

        if has_lh and has_rh:
            hip = (raw_kps[LH, :2] + raw_kps[RH, :2]) / 2
        elif has_lh:
            hip = raw_kps[LH, :2].copy()
        else:
            hip = raw_kps[RH, :2].copy()

        dy = hip[0] - sh[0]   # positive = hip below shoulder (correct for upright)
        dx = hip[1] - sh[1]   # horizontal deviation from vertical

        torso_len = math.sqrt(dy ** 2 + dx ** 2)
        if torso_len < 0.05 or dy < 0.05:
            return  # too short, horizontal, or ambiguous - skip

        # Camera tilt = angle of shoulder->hip vector from straight-down [0, +1]
        tilt = math.degrees(math.atan2(dx, dy))
        self._samples.append(tilt)
        self._frame_count += 1

        if not self.calibrated and len(self._samples) >= self.MIN_SAMPLES:
            self._commit(first=True)
        elif self.calibrated and self._frame_count % self.RECALIBRATE_INTERVAL == 0:
            self._commit(first=False)

    def _commit(self, first=False):
        samples = sorted(self._samples)
        self.tilt_deg = samples[len(samples) // 2]
        self.calibrated = True
        if first:
            print(f"[CALIB] Camera tilt auto-calibrated: {self.tilt_deg:+.1f}deg "
                  f"(from {len(self._samples)} samples)")

    @property
    def progress(self):
        """0.0-1.0: how far towards the first calibration commit."""
        return min(1.0, len(self._samples) / self.MIN_SAMPLES)

    @property
    def samples_needed(self):
        return max(0, self.MIN_SAMPLES - len(self._samples))


# Camera capture
class CameraStream:
    """
    Read frames in a background thread so the main loop never waits
    """

    def __init__(self, src=0, width=640, height=480, auto_exposure=True, exposure=95):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Disable auto white balance unconditionally - keeps colour consistent
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4600)  # neutral daylight WB

        if auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1=manual
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            print(f"  Camera: fixed exposure={self.cap.get(cv2.CAP_PROP_EXPOSURE):.0f}, auto_wb=off")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_GAIN, 0)
            print(f"  Camera: manual exposure={self.cap.get(cv2.CAP_PROP_EXPOSURE):.0f}, auto_wb=off")
        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


# Async wrapper: runs a model in a background thread so the main loop never blocks
class AsyncModel:
    """Submit a frame for inference; retrieve result on the next call.

    The main loop calls submit(frame) and gets the *previous* result back
    immediately - no waiting. One frame of latency, but main loop FPS is
    unaffected by inference time.
    """

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()
        self._result = None
        self._pending = False

    def run(self, frame):
        """Submit frame for inference. Returns last completed result (may be None)."""
        if not self._pending:
            self._pending = True
            frame_copy = frame.copy()
            threading.Thread(target=self._infer, args=(frame_copy,), daemon=True).start()
        with self._lock:
            return self._result

    def _infer(self, frame):
        kps = self._model.run(frame)
        with self._lock:
            self._result = kps
            self._pending = False


# Model (singlepose only)
class PoseModel:
    """Create singlepose MoveNet TFLite instance
    
    Returns:
        kps(np.array): Keypoints with shape (17, 3) as [y, x, confidence]
    """

    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path, num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Pre-compute model info
        input_shape = self.input_details[0]['shape']
        self.input_h = int(input_shape[1])
        self.input_w = int(input_shape[2])
        self.input_dtype = self.input_details[0]['dtype']
        self.input_index = self.input_details[0]['index']
        self.output_index = self.output_details[0]['index']

        # Pre-allocate input buffer
        self._input_buf = np.zeros((1, self.input_h, self.input_w, 3), dtype=self.input_dtype)

        dtype_name = self.input_dtype.__name__
        print(f"  Model: {model_path}")
        print(f"  Input: {self.input_h}x{self.input_w} ({dtype_name})")
        print(f"  Threads: 4")

    def run(self, frame):
        # Resize first (smaller data), then BGR->RGB
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if self.input_dtype == np.float32:
            np.copyto(self._input_buf[0], resized.astype(np.float32))
        else:
            # INT8/UINT8: pixel values already 0-255 uint8
            np.copyto(self._input_buf[0], resized)

        self.interpreter.set_tensor(self.input_index, self._input_buf)
        self.interpreter.invoke()

        kps = self.interpreter.get_tensor(self.output_index)
        kps = np.squeeze(kps)
        if kps.ndim == 3:
            kps = kps[0]
        return kps  # shape (17, 3)


# Feature extraction
def find_features(kps, thresh=0.4):
    """
    Extract features from the keypoints with shape:
        (17, 3) as [y, x, confidence]

    Returns:
        dict: A dict of pose metrics:
            -torso_angle (float): Angle of torso from vertical (degrees)
            -head_feet_angle (float): Angle between head and feet
            -aspect_ratio (float): Bounding box width/height ratio
            -body_y_diff (float): Y difference between upper and lower body
            -body_x_spread (float): X spread between upper and lower body
            -center_y (float): Y coordinate of bbox center (0-1)
            -normalized_cy (float): Y relative to bbox (0=top, 1=bottom)
            -bbox_height/width (float): Dimensions (0-1)
            -ankle_visible (bool): True if at least one ankle is seen
            -keypoint_count (int): Count above confidence threshold
            -shoulder/hip_conf (float): Minimum confidence scores
            -bbox (tuple): (x_min, y_min, x_max, y_max) coordinates
        None: If core keypoints are missing or too few
    """
    confs = kps[:, 2]

    # Need at least one shoulder and one hip
    has_ls = confs[LS] >= thresh
    has_rs = confs[RS] >= thresh
    has_lh = confs[LH] >= thresh
    has_rh = confs[RH] >= thresh

    if not (has_ls or has_rs) or not (has_lh or has_rh):
        return None

    # Shoulder midpoint: use both if available, else the visible one
    if has_ls and has_rs:
        sh_mid = (kps[LS, :2] + kps[RS, :2]) / 2
    elif has_ls:
        sh_mid = kps[LS, :2].copy()
    else:
        sh_mid = kps[RS, :2].copy()

    # Hip midpoint: same logic
    if has_lh and has_rh:
        hip_mid = (kps[LH, :2] + kps[RH, :2]) / 2
    elif has_lh:
        hip_mid = kps[LH, :2].copy()
    else:
        hip_mid = kps[RH, :2].copy()
    delta = hip_mid - sh_mid                     # [dy, dx]
    torso_angle = math.degrees(math.atan2(abs(delta[1]), abs(delta[0]) + 1e-6))

    # Head-to-feet angle
    head_mask = confs[HEAD_IDX] > thresh
    feet_mask = confs[FEET_IDX] > thresh
    head_feet_angle = None

    if head_mask.any() and feet_mask.any():
        head_avg = kps[HEAD_IDX[head_mask], :2].mean(axis=0)  
        feet_avg = kps[FEET_IDX[feet_mask], :2].mean(axis=0)
        hf_d = head_avg - feet_avg
        head_feet_angle = math.degrees(math.atan2(abs(hf_d[1]), abs(hf_d[0]) + 1e-6))

    # Upper-body vs lower-body comparison to catch horizontal poses
    upper_mask = confs[UPPER_BODY_IDX] > thresh
    lower_mask = confs[LOWER_BODY_IDX] > thresh
    body_y_diff = None
    body_x_spread = None

    if upper_mask.sum() >= 2 and lower_mask.sum() >= 2:
        upper_pts = kps[UPPER_BODY_IDX[upper_mask], :2]  
        lower_pts = kps[LOWER_BODY_IDX[lower_mask], :2]
        # Average Y of upper body vs lower body
        # Small difference = body is horizontal
        body_y_diff = abs(float(upper_pts[:, 0].mean() - lower_pts[:, 0].mean()))
        # X spread = how far apart upper and lower body are horizontally
        # Large spread = body is stretched out sideways
        body_x_spread = abs(float(upper_pts[:, 1].mean() - lower_pts[:, 1].mean()))

    # Bounding box from all visible keypoints 
    visible = confs >= thresh
    kp_count = visible.sum()
    if kp_count < 3:
        return None

    vis_pts = kps[visible, :2]  # shape (N, 2) as [y, x]
    ys = vis_pts[:, 0]
    xs = vis_pts[:, 1]

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    aspect_ratio = bbox_w / (bbox_h + 1e-6)
    center_y = ys.mean()
    norm_cy = (center_y - y_min) / (bbox_h + 1e-6)

    ankle_visible = confs[15] >= thresh or confs[16] >= thresh

    return {
        'torso_angle': torso_angle,
        'head_feet_angle': head_feet_angle,
        'aspect_ratio': float(aspect_ratio),
        'body_y_diff': body_y_diff,
        'body_x_spread': body_x_spread,
        'center_y': float(center_y),
        'normalized_cy': float(norm_cy),
        'bbox_height': float(bbox_h),
        'bbox_width': float(bbox_w),
        'ankle_visible': ankle_visible,
        'keypoint_count': int(kp_count),
        'shoulder_conf': float(min(confs[LS], confs[RS]) if (has_ls and has_rs) else max(confs[LS], confs[RS])),
        'hip_conf': float(min(confs[LH], confs[RH]) if (has_lh and has_rh) else max(confs[LH], confs[RH])),
        'bbox': (float(x_min), float(y_min), float(x_max), float(y_max)),
    }


def apply_tilt_correction(kps, tilt_deg):
    """Rotate keypoints to compensate for a tilted camera

    Args:
        kps: (17, 3) array of [y, x, confidence] in normalised [0,1] coords.
        tilt_deg: camera tilt in degrees; positive = camera rotated clockwise.
    Returns:
        corrected copy of kps (17, 3) - confidences unchanged.
    """
    if tilt_deg == 0.0:
        return kps
    theta = math.radians(-tilt_deg)   # counterclockwise compensation
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corrected = kps.copy()
    dy = kps[:, 0] - 0.5              # y relative to image centre
    dx = kps[:, 1] - 0.5              # x relative to image centre
    corrected[:, 0] = 0.5 + dy * cos_t - dx * sin_t
    corrected[:, 1] = 0.5 + dx * cos_t + dy * sin_t
    return corrected


def apply_elevation_correction(kps, elev_deg):
    """Stretch keypoint Y-coordinates to compensate for an elevated camera

    Args:
        kps: (17, 3) array of [y, x, confidence] in normalised [0,1] coords.
        elev_deg: camera elevation in degrees above horizontal (0=horizontal,
                  90=straight down). Values outside [5, 85] are clamped to
                  avoid division by ~zero.
    Returns:
        corrected copy of kps (17, 3) - confidences unchanged.
    """
    if elev_deg == 0.0:
        return kps
    elev_rad = math.radians(max(5.0, min(85.0, elev_deg)))
    scale_y = 1.0 / math.sin(elev_rad)
    corrected = kps.copy()
    corrected[:, 0] = 0.5 + (kps[:, 0] - 0.5) * scale_y
    return corrected


def is_upright(f, torso_t=35, ratio_t=1.0):
    return f['torso_angle'] < torso_t and f['aspect_ratio'] < ratio_t


def fall_score(f, torso_t=50, ratio_t=1.0, hf_t=55,
               velocity=0.0, vel_thresh=0.012,
               height_ratio=1.0, height_collapse=0.5,
               imu_score=0.0, cy_drop=0.0, camera_active=True):
    """Compute a weighted fall score 0.0-1.0 from all available signals.

    Each metric contributes a partial score based on how far it exceeds
    its threshold. This handles distance/angle variation because no single
    metric needs to fully trigger - several borderline signals combine.

    Returns:
        float: 0.0 = clearly upright, 1.0 = clearly fallen
    """
    s = 0.0

    # 1) Torso angle: 0-0.25 (how far past threshold)
    #    Ramps from 0 at torso_t*0.6 to 0.25 at torso_t*1.5
    torso_low = torso_t * 0.6
    torso_high = torso_t * 1.5
    if f['torso_angle'] > torso_low:
        s += 0.25 * min(1.0, (f['torso_angle'] - torso_low) / (torso_high - torso_low + 1e-6))

    # 2) Aspect ratio: 0-0.25
    #    Ramps from 0 at ratio_t*0.7 to 0.25 at ratio_t*2.0
    ratio_low = ratio_t * 0.7
    ratio_high = ratio_t * 2.0
    if f['aspect_ratio'] > ratio_low:
        s += 0.25 * min(1.0, (f['aspect_ratio'] - ratio_low) / (ratio_high - ratio_low + 1e-6))

    # 3) Head-feet angle: 0-0.2
    hf = f.get('head_feet_angle')
    if hf is not None:
        hf_low = hf_t * 0.6
        hf_high = hf_t * 1.5
        if hf > hf_low:
            s += 0.2 * min(1.0, (hf - hf_low) / (hf_high - hf_low + 1e-6))

    # 4) Body horizontal spread: 0-0.15
    yd = f.get('body_y_diff')
    xs = f.get('body_x_spread')
    if yd is not None and xs is not None:
        # Small y_diff + large x_spread = horizontal body
        horiz = 0.0
        if yd < 0.15:
            horiz += 0.5 * (1.0 - yd / 0.15)  # closer to 0 = more horizontal
        if xs > 0.1:
            horiz += 0.5 * min(1.0, xs / 0.5)
        s += 0.15 * min(1.0, horiz)

    # 5) Velocity (rapid drop): 0-0.20  (raised from 0.10 - key for parallel falls)
    if velocity > vel_thresh * 0.5:
        s += 0.20 * min(1.0, velocity / (vel_thresh * 2 + 1e-6))

    # 6) Height collapse: 0-0.20  (raised from 0.10 - key for parallel/side falls)
    if height_ratio < 1.0:
        collapse = 1.0 - height_ratio  # 0 = standing, 1 = zero height
        collapse_thresh = 1.0 - height_collapse  # 0.5
        if collapse > collapse_thresh * 0.5:
            s += 0.20 * min(1.0, collapse / (collapse_thresh + 1e-6))

    # 7) IMU boost: 0-0.15 normally, 0-0.25 when camera is actively tracking
    #    Higher weight when both camera and IMU agree — more reliable combined signal
    imu_weight = 0.25 if camera_active else 0.15
    s += imu_weight * imu_score

    # 8) Floor-level paradox: 0-0.50
    #    Person appears upright in 2D (torso/ratio metrics low) but their center of
    #    mass has dropped far below where it sat when they were genuinely standing.
    #    This catches face-toward-camera falls where foreshortening makes the body
    #    look vertical.  Guard: height_ratio > 0.7 excludes far-away perspective
    #    (person walking to far wall also drops in frame, but appears smaller).
    if cy_drop > 0.08 and height_ratio > 0.7:
        s += 0.50 * min(1.0, (cy_drop - 0.08) / 0.15)

    return min(1.0, s)


# Keep is_fallen as a convenience wrapper for the scoring system
def is_fallen(f, torso_t=50, ratio_t=1.0, hf_t=55, **kwargs):
    return fall_score(f, torso_t, ratio_t, hf_t, **kwargs) >= 0.42

# Fall state tracker
class FallTracker:
    # Grace period at startup before fall detection is active
    STARTUP_GRACE = 3.0

    # Max time to stay in potential_fall without confirming
    POTENTIAL_FALL_TIMEOUT = 5.0

    # Velocity detection: rapid downward movement of center of mass
    VELOCITY_WINDOW = 15        # frames to compute velocity over (~0.5s at 30fps)
    VELOCITY_THRESH = 0.012     # center_y rise per frame (normalized) = fast drop
    # Height baseline: detect sudden collapse relative to standing height
    HEIGHT_BASELINE_FRAMES = 60 # frames of upright data to establish baseline (~2s)
    HEIGHT_COLLAPSE_RATIO = 0.5 # bbox_height < 50% of baseline = collapsed

    def __init__(self, confirm_time=1.5, cooldown=5.0, startup_grace=None):
        self.state = 'unknown'
        self.last_upright_t = time.time()
        self.fall_start_t = 0.0
        self.last_alert_t = 0.0
        self.cy_history = deque(maxlen=30)
        self.last_features = None
        self.confirm_time = confirm_time
        self.cooldown = cooldown
        self.frames_missing = 0
        self.inference_gated = False   # set True by main loop when motion gate suppresses inference
        self._first_detection = True
        self._start_time = time.time()
        self._startup_grace = self.STARTUP_GRACE if startup_grace is None else startup_grace

        # Velocity tracking: timestamped center_y and bbox_height history
        self._cy_ts = deque(maxlen=30)      # (timestamp, center_y)
        self._bh_ts = deque(maxlen=30)      # (timestamp, bbox_height)
        self.velocity = 0.0                 # current center_y velocity (positive = falling)
        self.height_ratio = 1.0             # current bbox_height / baseline

        # Standing height baseline: learned from upright frames
        self._upright_heights = deque(maxlen=self.HEIGHT_BASELINE_FRAMES)
        self.standing_height = None         # established after enough samples
        self.current_fall_score = 0.0       # latest weighted fall score

        # Standing center_y baseline: where the person's center of mass sits in the
        # image while upright.  A large drop below this - even if the 2D skeleton
        # still looks "upright" - means the person is at floor level (face-down fall
        # toward the camera fools torso/ratio metrics but not center_y position).
        self._upright_cy = deque(maxlen=self.HEIGHT_BASELINE_FRAMES)
        self.baseline_center_y = None       # median center_y from upright frames
        self.cy_drop = 0.0                  # current center_y - baseline (positive = lower)

        # Sustained fall tracking: detect slow / angle-obscured falls that never
        # had a recent upright frame (transition_t gate would otherwise block them)
        self._fell_since = None             # time when fell score first exceeded threshold in unknown
        self._imu_triggered_fall = False    # True when fall was fired by IMU standalone (not camera)

        # Debug histories
        self.torso_hist = deque(maxlen=90)
        self.ratio_hist = deque(maxlen=90)
        self.hf_hist = deque(maxlen=90)
        self.velocity_hist = deque(maxlen=90)
        self.height_ratio_hist = deque(maxlen=90)
        self.fall_score_hist = deque(maxlen=90)

    # Tracking loss thresholds (in frames at ~30fps)
    MISSING_CONFIRM_FALL = 20    # ~0.7s: lost during potential_fall -> confirm fall
    MISSING_KEEP_FALLEN = 300    # ~10s: stay fallen even without keypoints
    MISSING_RESET_UNKNOWN = 90   # ~3s: lost in unknown state -> stay unknown (no action)

    def update(self, feat, now, torso_t, ratio_t, hf_t, transition_t=1.5, record_debug=False, imu_score=0.0):
        if feat is None:
            self.frames_missing += 1

            if self.state == 'potential_fall':
                # Person was falling and model lost them -> likely on the ground
                if self.frames_missing > self.MISSING_CONFIRM_FALL:
                    if (now - self.last_alert_t) > self.cooldown:
                        self.last_alert_t = now
                        self.state = 'fallen'
                        return 'fall'
                    self.state = 'fallen'

            elif self.state == 'fallen':
                # Stay fallen - horizontal person is hard to detect
                # Only reset after extended period (person may have left the scene)
                if self.frames_missing > self.MISSING_KEEP_FALLEN:
                    self.state = 'unknown'

            return None

        self.frames_missing = 0
        self.last_features = feat

        # On first valid detection refresh the upright timestamp
        if self._first_detection:
            self.last_upright_t = now
            self._first_detection = False

        upright_torso = torso_t * 0.7   # e.g. 50 * 0.7 = 35
        upright_ratio = ratio_t * 0.9   # e.g. 1.0 * 0.9 = 0.9
        up = is_upright(feat, upright_torso, upright_ratio)

        # --- Velocity tracking ---
        self._cy_ts.append((now, feat['center_y']))
        self._bh_ts.append((now, feat['bbox_height']))

        # Compute velocity: change in center_y over recent window
        self.velocity = 0.0
        if len(self._cy_ts) >= 2:
            oldest_t, oldest_cy = self._cy_ts[0]
            dt = now - oldest_t
            if dt > 0.1:  # need at least 0.1s of data
                # Positive velocity = center_y increasing = person moving down
                self.velocity = (feat['center_y'] - oldest_cy) / max(1, len(self._cy_ts))

        # --- Standing height baseline ---
        if up and feat['bbox_height'] > 0.1:
            self._upright_heights.append(feat['bbox_height'])
            if len(self._upright_heights) >= 10:
                sorted_h = sorted(self._upright_heights)
                self.standing_height = sorted_h[len(sorted_h) // 2]

            # Collect center_y baseline from the same upright frames
            self._upright_cy.append(feat['center_y'])
            if len(self._upright_cy) >= 10:
                sorted_cy = sorted(self._upright_cy)
                self.baseline_center_y = sorted_cy[len(sorted_cy) // 2]

        # Height ratio relative to baseline
        self.height_ratio = 1.0
        if self.standing_height and self.standing_height > 0.05:
            self.height_ratio = feat['bbox_height'] / self.standing_height

        # Center-y drop: positive = person's mass is lower in frame than when standing
        self.cy_drop = 0.0
        if self.baseline_center_y is not None:
            self.cy_drop = feat['center_y'] - self.baseline_center_y

        # --- Weighted fall score (all signals combined) ---
        self.current_fall_score = fall_score(
            feat, torso_t, ratio_t, hf_t,
            velocity=self.velocity, vel_thresh=self.VELOCITY_THRESH,
            height_ratio=self.height_ratio, height_collapse=self.HEIGHT_COLLAPSE_RATIO,
            imu_score=imu_score,
            cy_drop=self.cy_drop,
            camera_active=(feat is not None),
        )
        fell = self.current_fall_score >= 0.45

        if record_debug:
            self.torso_hist.append(feat['torso_angle'])
            self.ratio_hist.append(feat['aspect_ratio'])
            if feat.get('head_feet_angle') is not None:
                self.hf_hist.append(feat['head_feet_angle'])
            self.velocity_hist.append(self.velocity)
            self.height_ratio_hist.append(self.height_ratio)
            self.fall_score_hist.append(self.current_fall_score)

        self.cy_history.append(feat['normalized_cy'])

        if up:
            self.last_upright_t = now

        if (now - self._start_time) < self._startup_grace:
            return None

        event = None

        # Parallel/side fall trigger: strong velocity + height collapse even if
        # pose score hasn't crossed threshold yet (body edge-on to camera)
        motion_trigger = (
            self.velocity > self.VELOCITY_THRESH * 0.8
            and self.height_ratio < self.HEIGHT_COLLAPSE_RATIO * 1.3
            and self.standing_height is not None  # only after baseline is learned
        )

        if self.state == 'unknown':
            # IMU direct trigger: strong free-fall+impact reading bypasses
            # the pose wait entirely - IMU sees the fall event in real-time
            imu_direct = imu_score >= 0.70

            triggered = fell or motion_trigger or imu_direct

            # Track how long we've been seeing a fall pose in unknown state
            # (catches slow falls / people who were never seen upright)
            if triggered:
                if self._fell_since is None:
                    self._fell_since = now
            else:
                self._fell_since = None

            # Normal path: recent upright frame -> potential_fall
            if (now - self.last_upright_t) < transition_t and triggered:
                self.state = 'potential_fall'
                self.fall_start_t = now
                self._fell_since = None
                if imu_direct and not fell:
                    print(f"[TRACKER] IMU direct trigger (score={imu_score:.2f})")

            # Sustained path: score consistently high AND person was seen upright recently
            # (8s window catches slow falls; high score bar avoids slow-bend false positives)
            elif (
                self._fell_since is not None
                and (now - self._fell_since) > self.confirm_time
                and self.current_fall_score >= 0.65
                and (now - self.last_upright_t) < 8.0
            ):
                print(f"[TRACKER] Sustained fall ({now - self._fell_since:.1f}s, score={self.current_fall_score:.2f})")
                if (now - self.last_alert_t) > self.cooldown:
                    self.last_alert_t = now
                    event = 'fall'
                self.state = 'fallen'
                self._fell_since = None

        elif self.state == 'potential_fall':
            # IMU score reduces confirm time: 0.0 = no reduction, 1.0 = 50% faster
            imu_factor = 1.0 - (imu_score * 0.5)
            effective_confirm = self.confirm_time * imu_factor
            if fell:
                if (now - self.fall_start_t) > effective_confirm:
                    if (now - self.last_alert_t) > self.cooldown:
                        self.last_alert_t = now
                        event = 'fall'
                    self.state = 'fallen'
            elif up:
                self.state = 'unknown'
            else:
                if (now - self.fall_start_t) > self.POTENTIAL_FALL_TIMEOUT:
                    self.state = 'unknown'

        elif self.state == 'fallen':
            if up:
                self.state = 'unknown'
                self.cy_history.clear()
                self._fell_since = None
                self._imu_triggered_fall = False
                event = 'recovered'

        return event


# Drawing 
def draw_skeleton(frame, kps, thresh=0.4):
    """Draw skeleton and keypoints on the frame
    Args:
        -frame: The image frame to draw on (BGR format).
        -kps: Keypoints array of shape (17, 3) with [y, x, confidence].
        -thresh: Confidence threshold for drawing keypoints and connections.
    """
    h, w = frame.shape[:2]
    # Batch convert to pixel coords
    valid = kps[:, 2] > thresh
    px = (kps[:, 1] * w).astype(np.int32)
    py = (kps[:, 0] * h).astype(np.int32)

    # Draw connections
    for c in SKELETON_CONNECTIONS:
        i, j = c
        if valid[i] and valid[j]:
            cv2.line(frame, (px[i], py[i]), (px[j], py[j]), SKELETON_COLOR, 2)

    # Draw keypoints
    for k in range(17):
        if valid[k]:
            cv2.circle(frame, (px[k], py[k]), 4, SKELETON_COLOR, -1)


def draw_bbox(frame, feat):
    """Draw bounding box around detected person"""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = feat['bbox']
    cv2.rectangle(frame, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), SKELETON_COLOR, 1)


def draw_debug(frame, tracker, thresholds, socket_state=None, imu_analyzer=None, calibrator=None):
    """Render debug info into a separate side panel and return it."""
    h = frame.shape[0]
    pw = 260
    # Allocate a tall panel so content never gets cut off; crop to frame height at the end
    panel = np.zeros((1800, pw + 10, 3), dtype=np.uint8)

    feat = tracker.last_features
    x = 5
    y = 22
    F = cv2.FONT_HERSHEY_SIMPLEX
    W = (255,255,255)
    G = (150,150,150)

    # State
    sc = (0,0,255) if tracker.state in ('potential_fall','fallen') else (0,255,0)
    cv2.putText(panel, f"STATE: {tracker.state.upper()}", (x,y), F, 0.5, sc, 1)
    y += 22

    if feat is None:
        cv2.putText(panel, "No keypoints (need shoulders + hips)", (x,y), F, 0.35, G, 1)
        return panel[:h]

    tt, rt, ht = thresholds['torso'], thresholds['ratio'], thresholds['hf']

    def val_line(label, val, thresh, fmt=".1f"):
        nonlocal y
        over = val > thresh
        c = (0,0,255) if over else (0,255,0)
        txt = f"{label}: {val:{fmt}}  (thresh: {thresh:{fmt}})"
        cv2.putText(panel, txt, (x, y), F, 0.38, c, 1)
        # Mini bar
        bar_y = y + 3
        bar_w = pw - 10
        ratio = max(0, min(1, val / (thresh * 2 + 1e-6)))
        t_pos = int((thresh / (thresh * 2 + 1e-6)) * bar_w)
        cv2.rectangle(panel, (x, bar_y), (x + bar_w, bar_y + 6), (40,40,40), -1)
        cv2.rectangle(panel, (x, bar_y), (x + int(ratio * bar_w), bar_y + 6), c, -1)
        cv2.line(panel, (x + t_pos, bar_y - 1), (x + t_pos, bar_y + 7), (0,0,255), 1)
        y += 24

    val_line("Torso", feat['torso_angle'], tt)

    hf = feat.get('head_feet_angle')
    if hf is not None:
        val_line("Head-Feet", hf, ht)

    val_line("Ratio", feat['aspect_ratio'], rt, ".2f")

    # Upper and lower body comparison
    yd = feat.get('body_y_diff')
    xs = feat.get('body_x_spread')
    if yd is not None and xs is not None:
        # Show as inverted bar --> lower y_diff = more horizontal = more red
        y_diff_alert = yd <= 0.08 and xs > 0.3
        c = (0,0,255) if y_diff_alert else (0,255,0)
        cv2.putText(panel, f"UB-LB dy:{yd:.3f} dx:{xs:.3f}", (x, y), F, 0.35, c, 1)
        y += 18

    # Velocity and height baseline
    vel = tracker.velocity
    vel_c = (0, 0, 255) if vel > tracker.VELOCITY_THRESH else (0, 255, 0)
    cv2.putText(panel, f"Velocity: {vel:.4f}  (thresh: {tracker.VELOCITY_THRESH})", (x, y), F, 0.35, vel_c, 1)
    y += 16
    hr = tracker.height_ratio
    sh = tracker.standing_height
    hr_c = (0, 0, 255) if hr < tracker.HEIGHT_COLLAPSE_RATIO else (0, 255, 0)
    sh_txt = f"{sh:.3f}" if sh else "learning..."
    cv2.putText(panel, f"Ht ratio: {hr:.2f}  baseline: {sh_txt}", (x, y), F, 0.35, hr_c, 1)
    y += 16
    cy_drop = tracker.cy_drop
    bcy = tracker.baseline_center_y
    bcy_txt = f"{bcy:.3f}" if bcy is not None else "learning..."
    cy_c = (0, 0, 255) if cy_drop > 0.08 else (0, 255, 0)
    cv2.putText(panel, f"cy_drop: {cy_drop:+.3f}  base_cy: {bcy_txt}", (x, y), F, 0.35, cy_c, 1)
    y += 18

    y += 5
    cv2.putText(panel, f"keypoints: {feat['keypoint_count']}/17  "
                f"sh:{feat['shoulder_conf']:.2f} hp:{feat['hip_conf']:.2f}",
                (x, y), F, 0.33, G, 1)
    y += 16
    cv2.putText(panel, f"center_y: {feat['center_y']:.3f}  "
                f"norm_cy: {feat['normalized_cy']:.3f}",
                (x, y), F, 0.33, G, 1)
    y += 16
    cv2.putText(panel, f"bbox: {feat['bbox_width']:.2f}x{feat['bbox_height']:.2f}  "
                f"ankle: {'Y' if feat['ankle_visible'] else 'N'}",
                (x, y), F, 0.33, G, 1)
    y += 20

    up = is_upright(feat)
    fs = tracker.current_fall_score
    fell = fs >= 0.45
    fs_c = (0,0,255) if fs >= 0.45 else (0,165,255) if fs >= 0.3 else (0,255,0)
    cv2.putText(panel, f"upright={up}  score={fs:.2f}", (x, y), F, 0.4, fs_c, 1)
    y += 4
    # Score bar with threshold marker
    bar_w = pw - 10
    cv2.rectangle(panel, (x, y), (x + bar_w, y + 8), (40, 40, 40), -1)
    cv2.rectangle(panel, (x, y), (x + int(fs * bar_w), y + 8), fs_c, -1)
    t_px = x + int(0.45 * bar_w)
    cv2.line(panel, (t_px, y - 1), (t_px, y + 9), (0, 0, 255), 1)
    y += 16

    # Mini graphs
    def mini_graph(label, data, vmin, vmax, thresh, color):
        nonlocal y
        if len(data) < 2:
            return
        cv2.putText(panel, label, (x, y), F, 0.33, G, 1)
        y += 3
        gw, gh = pw - 10, 28
        cv2.rectangle(panel, (x, y), (x+gw, y+gh), (30,30,30), -1)
        # Threshold line
        tr = max(0, min(1, (thresh - vmin) / (vmax - vmin + 1e-6)))
        cv2.line(panel, (x, y + gh - int(tr*gh)), (x+gw, y + gh - int(tr*gh)), (0,0,255), 1)
        # Data line - sample every other point for speed
        pts = list(data)
        n = len(pts)
        step = max(1, n // 60)  # limit to ~60 points
        prev = None
        for i in range(0, n, step):
            r = max(0, min(1, (pts[i] - vmin) / (vmax - vmin + 1e-6)))
            px_pt = x + int(i / (n - 1 + 1e-6) * gw)
            py_pt = y + gh - int(r * gh)
            if prev is not None:
                cv2.line(panel, prev, (px_pt, py_pt), color, 1)
            prev = (px_pt, py_pt)
        y += gh + 8

    mini_graph("torso", tracker.torso_hist, 0, 90, tt, (0,200,255))
    mini_graph("ratio", tracker.ratio_hist, 0, 3, rt, (255,200,0))
    mini_graph("hd-ft", tracker.hf_hist, 0, 90, ht, (200,0,255))
    mini_graph("velocity", tracker.velocity_hist, -0.02, 0.04,
               tracker.VELOCITY_THRESH, (255, 100, 100))
    mini_graph("ht_ratio", tracker.height_ratio_hist, 0, 1.5,
               tracker.HEIGHT_COLLAPSE_RATIO, (100, 255, 100))
    mini_graph("fall_score", tracker.fall_score_hist, 0, 1.0,
               0.45, (0, 200, 255))

    # Socket status section
    if socket_state is not None:
        y += 6
        cv2.line(panel, (x, y), (x + pw - 10, y), (60, 60, 60), 1)
        y += 10
        snap = socket_state.snapshot()
        conn_c = (0, 255, 0) if snap['connected'] else (80, 80, 80)
        conn_txt = "WS: CONNECTED" if snap['connected'] else "WS: DISCONNECTED"
        cv2.putText(panel, conn_txt, (x, y), F, 0.38, conn_c, 1)
        y += 16

        res = snap['last_result']
        if res is None:
            res_txt = "Phone: --"
            res_c = G
        else:
            ago = time.time() - snap['last_msg_t']
            res_txt = f"Phone: {res}  ({ago:.0f}s ago)"
            res_c = (0, 0, 255) if res == 'fall' else (0, 200, 100)
        cv2.putText(panel, res_txt, (x, y), F, 0.38, res_c, 1)
        y += 16

        # IMU score display
        if imu_analyzer is not None:
            imu_s = imu_analyzer.get_hint()
            if imu_s > 0:
                reduction = int(imu_s * 50)
                imu_c = (0, 0, 255) if imu_s >= 0.6 else (0, 165, 255) if imu_s >= 0.3 else (0, 200, 100)
                imu_txt = f"IMU score: {imu_s:.2f} (confirm -{reduction}%)"
            else:
                imu_c = G
                imu_txt = "IMU score: no data"
            cv2.putText(panel, imu_txt, (x, y), F, 0.35, imu_c, 1)
            y += 16
            tilt = imu_analyzer.last_tilt
            cal = imu_analyzer._baseline_gravity is not None
            tilt_c = (0, 0, 255) if tilt > 75.0 else (0, 165, 255) if tilt > 60.0 else (0, 200, 100)
            cal_str = "cal" if cal else "uncal"
            cv2.putText(panel, f"IMU tilt: {tilt:.1f} deg ({cal_str})", (x, y), F, 0.35, tilt_c, 1)
            y += 16

            # IMU breakdown bar
            if imu_s > 0:
                bar_w = pw - 10
                bar_y = y
                cv2.rectangle(panel, (x, bar_y), (x + bar_w, bar_y + 8), (40, 40, 40), -1)
                fill = int(imu_s * bar_w)
                cv2.rectangle(panel, (x, bar_y), (x + fill, bar_y + 8), imu_c, -1)
                # Threshold markers at 0.5 and 0.6
                for t_val in (0.5, 0.6):
                    t_px = x + int(t_val * bar_w)
                    cv2.line(panel, (t_px, bar_y - 1), (t_px, bar_y + 9), (0, 0, 255), 1)
                y += 14
        else:
            hint_active = snap.get('hint', False)
            hint_c = (0, 165, 255) if hint_active else G
            hint_txt = "WS hint: ACTIVE" if hint_active else "WS hint: inactive"
            cv2.putText(panel, hint_txt, (x, y), F, 0.35, hint_c, 1)
            y += 16

        # CSV preview
        csv_rows = snap.get('csv_preview', [])
        if csv_rows:
            y += 4
            csv_url = snap.get('csv_url', '')
            # Show truncated URL
            url_short = csv_url[-35:] if len(csv_url) > 35 else csv_url
            cv2.putText(panel, f"CSV: ...{url_short}", (x, y), F, 0.28, (180, 180, 100), 1)
            y += 13
            for i, row in enumerate(csv_rows):
                # Truncate long rows to fit panel
                display_row = row[:50] + "..." if len(row) > 50 else row
                cv2.putText(panel, f"{i}: {display_row}", (x, y), F, 0.28, (140, 140, 140), 1)
                y += 11

    # Calibration status
    if calibrator is not None:
        y += 6
        cv2.line(panel, (x, y), (x + pw - 10, y), (60, 60, 60), 1)
        y += 10
        if calibrator.calibrated:
            calib_c = (0, 255, 180)
            calib_txt = f"TILT CAL: {calibrator.tilt_deg:+.1f} deg (auto)"
        else:
            prog = int(calibrator.progress * (pw - 10))
            calib_c = (0, 165, 255)
            calib_txt = f"Calibrating... {calibrator.MIN_SAMPLES - calibrator.samples_needed}/{calibrator.MIN_SAMPLES}"
            cv2.rectangle(panel, (x, y + 3), (x + pw - 10, y + 10), (40, 40, 40), -1)
            cv2.rectangle(panel, (x, y + 3), (x + prog, y + 10), (0, 165, 255), -1)
            y += 12
        cv2.putText(panel, calib_txt, (x, y), F, 0.35, calib_c, 1)
        y += 14

    # Return only the filled portion (y + small margin), not the full 1800px allocation
    return panel[:min(y + 20, 1800)]


# WebSocket background thread
def start_ws_background(args, socket_state, cam_tracker, imu_analyzer=None, ipc=None):
    """Start a WebSocket client in a daemon thread.

    Receives CSV URLs from the phone and runs fall detection on them.
    When the camera is unavailable the IMU path fires fall events directly -
    IPC to the peripheral controller and notification to the phone - so the
    system works with camera, IMU, or both independently.
    """
    _last_imu_alert_t = [0.0]  # closure-safe cooldown for IMU-standalone alerts

    def on_open(ws):
        socket_state.set_connected(True)
        socket_state.set_ws(ws)
        print("[WS] Connected")

    def on_message(ws, message):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            print("[WS] Invalid JSON received")
            return
        url = data.get('data')
        if not url:
            return
        print(f"[WS] Received CSV URL: {url}")
        # Fetch CSV once, grab preview rows, then process
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                csv_content = resp.read().decode('utf-8')
        except Exception as e:
            print(f"[WS] Error fetching CSV: {e}")
            socket_state.set_result('error')
            return

        # Store first few rows as preview for debug panel
        preview_rows = csv_content.splitlines()[:SocketState.CSV_PREVIEW_LINES]
        socket_state.set_csv_preview(url, preview_rows)

        # Detect CSV type: IMU (6 cols: ax,ay,az,gx,gy,gz) vs keypoints (51 cols)
        first_data_row = None
        for line in csv_content.splitlines():
            try:
                cols = [float(v) for v in line.split(',')]
                first_data_row = cols
                break
            except ValueError:
                continue

        if first_data_row and len(first_data_row) == 6 and imu_analyzer:
            # IMU data - primary OR standalone fall detection
            score = imu_analyzer.analyze(csv_content)
            phone_result = 'fall' if score >= 0.5 else 'no_fall'
            print(f"[WS] IMU analysis: score={score:.2f} -> {phone_result}")

            # Fire fall event directly when IMU detects a fall, regardless of camera.
            # This mirrors the camera fall path so buzzer/LED/phone all activate
            # even when the camera is blocked, offline, or sees nothing.
            now = time.time()
            # Camera is not providing fresh data when:
            #   a) covered/blocked: no keypoints for >0.5s
            #   b) inference gated: motion threshold suppressed the model run
            camera_blind = (
                cam_tracker.frames_missing > 15
                or cam_tracker.inference_gated
            )
            imu_fall = (
                score >= 0.5
                and cam_tracker.state != 'fallen'
                and (now - _last_imu_alert_t[0]) > cam_tracker.cooldown
            )
            # Aggressive tilt trigger: fires when camera is blind (covered or gated)
            # tilt alone isn't reliable enough when the camera is actively tracking
            tilt_trigger = (
                imu_analyzer.last_tilt > 60.0
                and imu_analyzer._baseline_gravity is not None
                and camera_blind
                and cam_tracker.state != 'fallen'
                and (now - _last_imu_alert_t[0]) > cam_tracker.cooldown
            )
            if imu_fall or tilt_trigger:
                reason = f"tilt={imu_analyzer.last_tilt:.1f}deg" if (tilt_trigger and not imu_fall) else f"score={score:.2f}"
                _last_imu_alert_t[0] = now
                cam_tracker.state = 'fallen'
                cam_tracker.last_alert_t = now
                cam_tracker._imu_triggered_fall = True
                print(f"[WS] IMU standalone fall detected ({reason})")
                if ipc:
                    ipc.send({'type': 'event', 'event': 'fall_detected', 'ts': now})
                socket_state.send(json.dumps({'target': 'phone', 'data': ['fall']}))

            # IMU tilt reset: if this fall was IMU-triggered and the person's
            # motion has returned to normal (low score on new reading), recover
            elif (
                cam_tracker.state == 'fallen'
                and cam_tracker._imu_triggered_fall
                and score < 0.15
            ):
                cam_tracker.state = 'unknown'
                cam_tracker._imu_triggered_fall = False
                cam_tracker._fell_since = None
                cam_tracker.cy_history.clear()
                print(f"[WS] IMU tilt reset — fall state cleared (score={score:.2f})")
                if ipc:
                    ipc.send({'type': 'event', 'event': 'recovered', 'ts': now})
                socket_state.send(json.dumps({'target': 'phone', 'data': ['cancel']}))
        else:
            # Keypoint data
            try:
                phone_result = process_csv(csv_content, args, raw=True)
            except Exception as e:
                print(f"[WS] Error processing CSV: {e}")
                phone_result = 'error'
            print(f"[WS] Phone analysis: {phone_result}")

        socket_state.set_result(phone_result)

        # Reply to phone with combined camera + IMU status
        cam_fallen = cam_tracker.state in ('fallen', 'potential_fall')
        camera_result = 'fall' if cam_fallen else 'no_fall'
        print(f"[WS] Camera status: {camera_result}")
        ws.send(json.dumps({'target': 'phone', 'data': [camera_result]}))

    def on_error(_ws, error):
        print(f"[WS] Error: {error}")

    def on_close(_ws, _code, _msg):
        socket_state.set_connected(False)
        socket_state.set_ws(None)
        print("[WS] Disconnected")

    def _run():
        ws = websocket.WebSocketApp(
            args.ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever(reconnect=5)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# CSV processing (used by ws background thread)
def process_csv(url_or_content, args, raw=False):
    """Process a CSV of keypoints and run fall detection.

    Expected format: one row per frame, 51 columns [y0,x0,c0, y1,x1,c1, ..., y16,x16,c16].
    If raw=True, url_or_content is the CSV string directly. Otherwise it's a URL to fetch.
    Returns 'fall' if a fall event is detected otherwise 'no_fall'
    """
    if raw:
        content = url_or_content
    else:
        with urllib.request.urlopen(url_or_content, timeout=15) as resp:
            content = resp.read().decode('utf-8')

    reader = csv.reader(io.StringIO(content))
    # startup_grace=0: no warm-up silence for offline clips
    tracker = FallTracker(confirm_time=args.confirm_time, startup_grace=0.0)
    result = 'no_fall'
    now = time.time()

    for row in reader:
        if len(row) < 51:
            continue
        try:
            vals = [float(v) for v in row[:51]]
        except ValueError:
            continue  # skip header or malformed rows

        kps = np.array(vals, dtype=np.float32).reshape(17, 3)
        kps = apply_tilt_correction(kps, getattr(args, 'camera_tilt', 0.0))
        kps = apply_elevation_correction(kps, getattr(args, 'camera_elevation', 0.0))
        features = find_features(kps, args.score_thresh)
        # transition_t=9999: don't require prior upright state - clip may start mid-fall
        ev = tracker.update(features, now, args.torso_thresh, args.ratio_thresh, args.hf_thresh,
                            transition_t=9999)
        now += 0.033  # simulate ~30 fps timestamps
        if ev == 'fall':
            result = 'fall'
            break

    return result

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Fall Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Pose_Estimation.py --model movenet_lightning.tflite --debug
  python Pose_Estimation.py --skip-frames 2 --debug
  python Pose_Estimation.py --torso-thresh 45 --ratio-thresh 0.9 --hf-thresh 50
        """)
    parser.add_argument("--model", default="movenet_lightning_int8.tflite")
    parser.add_argument("--thunder-model", default="movenet_thunder_int8.tflite",
                        help="Thunder model path for high-accuracy fallback (set empty to disable)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=320, help="Camera capture width (default: 320)")
    parser.add_argument("--height", type=int, default=240, help="Camera capture height (default: 240)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=1, help="Run inference every N frames (1=every frame, 2=every other, etc)")
    parser.add_argument("--torso-thresh", type=float, default=50)
    parser.add_argument("--ratio-thresh", type=float, default=1.0)
    parser.add_argument("--hf-thresh", type=float, default=55)
    parser.add_argument("--confirm-time", type=float, default=1.5)
    parser.add_argument("--score-thresh", type=float, default=0.4, help="Keypoint confidence threshold (0.4 recommended for Lightning, Alam et al. used 0.5 for Thunder)")
    parser.add_argument("--camera-tilt", type=float, default=0.0,
                        help="Camera rotation in degrees (positive=clockwise). "
                             "Keypoints are counter-rotated before feature extraction "
                             "so a tilted camera does not bias angle metrics.")
    parser.add_argument("--camera-elevation", type=float, default=0.0,
                        help="Camera elevation in degrees above horizontal (0=eye-level, 90=straight down). "
                             "Keypoint Y-coordinates are stretched by 1/sin(elev) to undo the "
                             "vertical foreshortening caused by a high-mounted camera.")
    parser.add_argument("--no-auto-exposure", action="store_true", help="Lock camera to manual exposure instead of auto")
    parser.add_argument("--exposure", type=int, default=95,
                        help="Exposure value (default: 95, lower = darker). Always applied - auto WB is always off.")
    parser.add_argument("--no-auto-calibrate", action="store_true",
                        help="Disable auto-calibration of camera tilt; use --camera-tilt value directly")
    parser.add_argument("--motion-thresh", type=int, default=800,
                        help="Min foreground pixels to trigger inference (0=always run, default: 800)")
    parser.add_argument("--roi-pad", type=float, default=0.25,
                        help="Fractional padding around person bbox for ROI crop (default: 0.25)")
    parser.add_argument("--ipc", action="store_true", help="Enable IPC server for peripheral controller")
    parser.add_argument("--ipc-sock", default="/tmp/falldetect.sock", help="Unix socket path for IPC")
    parser.add_argument("--ws", action="store_true", help="Run in WebSocket hub mode (receive CSV, process, reply)")
    parser.add_argument("--ws-url", default="ws://localhost:8080/ws?id=hub", help="WebSocket endpoint URL")
    parser.add_argument("--upload-url", default="unabsorbing-perla-subsequently.ngrok-free.dev",
                        help="Base URL to upload snapshots to (default: ngrok endpoint)")
    parser.add_argument("--no-snapshot-server", action="store_true",
                        help="Disable snapshot upload (no pic messages sent to phone)")
    args = parser.parse_args()

    thresholds = {'torso': args.torso_thresh, 'ratio': args.ratio_thresh, 'hf': args.hf_thresh}

    tracker = FallTracker(confirm_time=args.confirm_time)
    calibrator = CameraCalibrator()

    # Snapshot server: captures frames and serves them for phone display
    snap = None
    if not args.no_snapshot_server:
        snap = SnapshotServer(base_url=args.upload_url)
        snap.start()

    # Shared frame reference - updated each iteration so IPC commands can grab it
    _current_frame = [None]

    ipc = None
    if args.ipc:
        ipc = IPCServer(args.ipc_sock)

        def on_ipc_command(msg):
            cmd = msg.get('cmd')

            if cmd == 'reset_fall':
                tracker.state = 'unknown'
                tracker.cy_history.clear()
                tracker._fell_since = None
                print("[IPC] Fall state reset")

            elif cmd == 'send_alert':
                # Voice said "call help" - push fall alert to phone
                if socket_state:
                    socket_state.send(json.dumps({'target': 'phone', 'data': ['fall']}))
                print("[IPC] Alert sent to phone via voice command")

            elif cmd == 'cancel_alert':
                # Voice said "false alarm" - tell phone to dismiss
                if socket_state:
                    socket_state.send(json.dumps({'target': 'phone', 'data': ['cancel']}))
                tracker.state = 'unknown'
                tracker.cy_history.clear()
                tracker._fell_since = None
                print("[IPC] Alert cancelled, phone notified")

            elif cmd == 'send_photo':
                # Voice said "take photo" - capture current frame and send URL to phone
                frame = _current_frame[0]
                if frame is not None and snap is not None and socket_state:
                    url = snap.capture(frame)
                    socket_state.send(json.dumps({'target': 'phone', 'data': ['pic', url]}))
                    print(f"[IPC] Photo sent to phone: {url}")
                elif not socket_state:
                    print("[IPC] send_photo: WebSocket not active (run with --ws)")
                elif snap is None:
                    print("[IPC] send_photo: snapshot server disabled")

        ipc.set_command_callback(on_ipc_command)
        ipc.start()
        print(f"[IPC] Listening on {args.ipc_sock}")

    socket_state = None
    imu_analyzer = None
    if args.ws:
        socket_state = SocketState()
        imu_analyzer = IMUAnalyzer()
        print(f"[WS] Connecting to {args.ws_url} in background ...")
        start_ws_background(args, socket_state, tracker, imu_analyzer, ipc=ipc)

    print("=" * 50)
    print("  Fall Detection System ")
    print("=" * 50)
    model = PoseModel(args.model)
    thunder = None
    if args.thunder_model and os.path.exists(args.thunder_model):
        thunder = AsyncModel(PoseModel(args.thunder_model))
        print(f"  Thunder cascade: ENABLED async (activates on potential_fall, non-blocking)")
    else:
        print(f"  Thunder cascade: disabled")
    print(f"  Camera: {args.camera} ({args.width}x{args.height}) | Skip: every {args.skip_frames} frame(s)")
    if args.no_auto_calibrate:
        print(f"  Tilt: manual {args.camera_tilt:+.1f}deg  Elevation: {args.camera_elevation:.1f}deg")
    else:
        print(f"  Tilt: AUTO-CALIBRATE (collecting {CameraCalibrator.MIN_SAMPLES} upright frames)")
        if args.camera_tilt != 0.0:
            print(f"  Note: --camera-tilt ignored while auto-calibration is active (use --no-auto-calibrate)")
    print(f"  Thresholds: torso>{args.torso_thresh} ratio>{args.ratio_thresh} hf>{args.hf_thresh}")
    print(f"  Confidence: {args.score_thresh}")
    print(f"  Debug: {'ON' if args.debug else 'OFF (press d to toggle)'}")
    print(f"  Press 'q' to quit, 'd' to toggle debug\n")

    cam = CameraStream(args.camera, width=args.width, height=args.height,
                       auto_exposure=not args.no_auto_exposure, exposure=args.exposure)
    if not cam.is_opened():
        raise RuntimeError("Cannot open camera")

    # Wait for first frame
    time.sleep(0.5)
    show_debug = args.debug
    _debug_window_open = False

    fps_time = time.time()
    fps_count = 0
    fps_val = 0
    inf_ms = 0
    frame_num = 0

    # Cache last inference results for skip-frame mode
    last_kps = None
    last_feat = None
    use_thunder = False

    # Motion gating: skip inference when room is still
    _mog = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    motion_px = 0  # foreground pixel count from last check

    # ROI tracking: crop to person's last known location
    last_roi = None  # (x1, y1, x2, y2) normalised [0,1] or None

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        _current_frame[0] = frame  # keep latest frame for snapshot on demand

        now = time.time()
        frame_num += 1

        # Run inference every Nth frame
        run_now = (frame_num % args.skip_frames == 0)

        if run_now:
            t0 = time.time()

            # --- Motion gating ---
            # Only run background subtraction every inference frame (not every frame)
            fg_mask = _mog.apply(frame)
            motion_px = int(np.count_nonzero(fg_mask))
            # Always run inference when tracking an alert state - never gate those
            skip_inference = (
                args.motion_thresh > 0
                and motion_px < args.motion_thresh
                and tracker.state == 'unknown'
            )

            tracker.inference_gated = skip_inference

            if not skip_inference:
                # --- ROI cropping ---
                # If we have a known person bbox, crop to that region so the
                # model sees a close-up rather than a tiny figure in a large frame.
                # Remap output keypoints back to full-frame coords afterwards.
                roi_crop = None
                if last_roi is not None:
                    h_f, w_f = frame.shape[:2]
                    pad = args.roi_pad
                    x1n, y1n, x2n, y2n = last_roi
                    # Add padding and clamp to [0, 1]
                    x1p = max(0.0, x1n - pad * (x2n - x1n))
                    y1p = max(0.0, y1n - pad * (y2n - y1n))
                    x2p = min(1.0, x2n + pad * (x2n - x1n))
                    y2p = min(1.0, y2n + pad * (y2n - y1n))
                    # Only crop if region is meaningfully smaller than full frame
                    if (x2p - x1p) < 0.85 or (y2p - y1p) < 0.85:
                        x1px = int(x1p * w_f)
                        y1px = int(y1p * h_f)
                        x2px = int(x2p * w_f)
                        y2px = int(y2p * h_f)
                        roi_crop = frame[y1px:y2px, x1px:x2px]
                        roi_crop = (roi_crop, x1p, y1p, x2p, y2p)

                # Cascade: use Thunder when state is uncertain, Lightning otherwise
                use_thunder = (
                    thunder is not None
                    and tracker.state in ('potential_fall', 'fallen')
                )
                active_model = thunder if use_thunder else model

                if roi_crop is not None:
                    crop_img, rx1, ry1, rx2, ry2 = roi_crop
                    raw_kps = active_model.run(crop_img)
                    # Remap normalised keypoint coords from crop -> full frame
                    if raw_kps is not None:
                        last_kps = raw_kps.copy()
                        last_kps[:, 0] = ry1 + raw_kps[:, 0] * (ry2 - ry1)
                        last_kps[:, 1] = rx1 + raw_kps[:, 1] * (rx2 - rx1)
                    else:
                        last_kps = raw_kps
                else:
                    last_kps = active_model.run(frame)

                inf_ms = (time.time() - t0) * 1000

                # Auto-calibrate tilt from raw keypoints when person looks upright
                if (last_kps is not None and not args.no_auto_calibrate):
                    raw_feat_check = find_features(last_kps, args.score_thresh)
                    if raw_feat_check is not None and raw_feat_check['aspect_ratio'] < 0.7:
                        calibrator.feed(last_kps, args.score_thresh)

                # Effective tilt: prefer auto-calibrated value unless manually specified
                # or auto-calibration is disabled
                if not args.no_auto_calibrate and calibrator.calibrated:
                    effective_tilt = calibrator.tilt_deg
                else:
                    effective_tilt = args.camera_tilt

                # Corrected copy for feature extraction --> last_kps stays raw for drawing
                if last_kps is not None:
                    kps_for_feat = apply_tilt_correction(last_kps, effective_tilt)
                    kps_for_feat = apply_elevation_correction(kps_for_feat, args.camera_elevation)
                    feat_thresh = args.score_thresh * (1.1 if use_thunder else 1.0)
                    last_feat = find_features(kps_for_feat, feat_thresh)

                    # Retry with lower confidence when tracking a fall
                    if last_feat is None and tracker.state in ('potential_fall', 'fallen'):
                        last_feat = find_features(kps_for_feat, args.score_thresh * 0.5)
                else:
                    last_feat = None

                # Update ROI for next frame from current bbox
                if last_feat is not None:
                    last_roi = last_feat['bbox']   # (x_min, y_min, x_max, y_max)
                else:
                    last_roi = None

            cur_imu_score = imu_analyzer.get_hint() if imu_analyzer else 0.0
            ev = tracker.update(last_feat, now, args.torso_thresh,
                                args.ratio_thresh, args.hf_thresh,
                                record_debug=show_debug, imu_score=cur_imu_score)
            if ev == 'fall':
                print(f"[{time.strftime('%H:%M:%S')}] FALL DETECTED")
                if ipc:
                    ipc.send({'type': 'event', 'event': 'fall_detected', 'ts': now})
                if socket_state:
                    socket_state.send(json.dumps({'target': 'phone', 'data': ['fall']}))
                    # Auto-send snapshot so phone displays the fall scene
                    if snap is not None and _current_frame[0] is not None:
                        def _send_snap(f=_current_frame[0]):
                            url = snap.capture(f)
                            socket_state.send(json.dumps({'target': 'phone', 'data': ['pic', url]}))
                        threading.Thread(target=_send_snap, daemon=True).start()
            elif ev == 'recovered':
                print(f"[{time.strftime('%H:%M:%S')}] Person recovered")
                if ipc:
                    ipc.send({'type': 'event', 'event': 'recovered', 'ts': now})

            # IPC status heartbeat ~every 2s at 30fps
            if ipc and frame_num % 60 == 0:
                ipc.send({'type': 'status', 'state': tracker.state, 'ts': now})

        # Headless mode
        if args.headless:
            if run_now and frame_num % 30 == 0:
                # Print periodic status
                if last_feat:
                    hf = last_feat.get('head_feet_angle')
                    hs = f"{hf:.0f}" if hf else "?"
                    yd = last_feat.get('body_y_diff')
                    yds = f"{yd:.3f}" if yd is not None else "?"
                    print(f"[{time.strftime('%H:%M:%S')}] {tracker.state} | "
                          f"torso={last_feat['torso_angle']:.0f} hf={hs} "
                          f"ratio={last_feat['aspect_ratio']:.1f} dy={yds} | {inf_ms:.0f}ms")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No body detected | {inf_ms:.0f}ms")
            time.sleep(0.01)
            continue

        # Draw skeleton
        if last_kps is not None:
            draw_skeleton(frame, last_kps, args.score_thresh)

        # Draw bounding box (only when features are valid)
        if last_feat is not None:
            draw_bbox(frame, last_feat)

        # Status
        if tracker.state == 'fallen':
            st, sc = "FALL DETECTED", (0,0,255)
        elif tracker.state == 'potential_fall':
            st, sc = "POSSIBLE FALL", (0,165,255)
        else:
            st, sc = "OK", (0,200,0)
        cv2.putText(frame, st, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sc, 3)

        # Compact info (when debug is off)
        if not show_debug:
            iy = 55
            if last_feat:
                hf = last_feat.get('head_feet_angle')
                hs = f"{hf:.0f}" if hf else "?"
                cv2.putText(frame, f"torso={last_feat['torso_angle']:.0f} hf={hs} "
                            f"ratio={last_feat['aspect_ratio']:.1f} [{tracker.state}]",
                            (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)
            else:
                cv2.putText(frame, "No body detected (need shoulders + hips visible)",
                            (10, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)

        # FPS
        fps_count += 1
        if now - fps_time > 1:
            fps_val = fps_count / (now - fps_time)
            fps_count = 0
            fps_time = now
        model_tag = "THN" if use_thunder else "LGT"
        gated_tag = " GATED" if (args.motion_thresh > 0 and motion_px < args.motion_thresh and tracker.state == 'unknown') else ""
        cv2.putText(frame, f"FPS:{fps_val:.0f} inf:{inf_ms:.0f}ms [{model_tag}] mot:{motion_px}{gated_tag}",
                    (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        # Scale main frame up for easier viewing (inference still at native res)
        display = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3))

        # Debug panel in its own window so proportions stay correct
        if show_debug:
            debug_panel = draw_debug(frame, tracker, thresholds, socket_state, imu_analyzer, calibrator)
            cv2.imshow('Debug', debug_panel)
            _debug_window_open = True
        elif _debug_window_open:
            cv2.destroyWindow('Debug')
            _debug_window_open = False

        cv2.imshow('Fall Detection', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug: {'ON' if show_debug else 'OFF'}")

    cam.release()
    if ipc:
        ipc.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()