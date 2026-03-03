"""
Usage:
  python Pose_Estimation.py --model movenet_lightning.tflite --debug
  python Pose_Estimation.py --skip-frames 2  # run model every 2nd frame; can do more at the cost of lag

Tuning:
  Threshold tuning example to increase sensitivity: python Pose_Estimation.py --torso-thresh 45 --ratio-thresh 0.9 --hf-thresh 50 

"""

import time
import argparse
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


# Camera capture
class CameraStream:
    """
    Read frames in a background thread so the main loop never waits
    """

    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
        # Convert BGR to RGB, resize, and copy to input buffer
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img, (self.input_w, self.input_h))
        np.copyto(self._input_buf[0], resized)

        self.interpreter.set_tensor(self.input_index, self._input_buf)
        self.interpreter.invoke()

        kps = self.interpreter.get_tensor(self.output_index)
        kps = np.squeeze(kps)
        if kps.ndim == 3:
            kps = kps[0]
        return kps  # shape (17, 3)


# Feature extraction
def find_features(kps, thresh=0.3):
    """
    Extract features from the keypoints with shape:
        (17, 3) as [y, x, confidence]
    Returns:
        dict: A dict of pose metrics:
            -torso_angle (float): Angle of torso from vertical (degrees).
            -head_feet_angle (float): Angle between head and feet.
            -aspect_ratio (float): Bounding box width/height ratio.
            -center_y (float): Y coordinate of bbox center (0-1).
            -normalized_cy (float): Y relative to bbox (0=top, 1=bottom).
            -bbox_height/width (float): Dimensions (0-1).
            -ankle_visible (bool): True if at least one ankle is seen.
            -keypoint_count (int): Count above confidence threshold.
            -shoulder/hip_conf (float): Minimum confidence scores.
            -bbox (tuple): (x_min, y_min, x_max, y_max) coordinates.
        None: If core keypoints are missing or too few.
    """
    confs = kps[:, 2]

    # Check core keypoints: both shoulders, both hips
    if confs[LS] < thresh or confs[RS] < thresh or confs[LH] < thresh or confs[RH] < thresh:
        return None

    # Torso angle
    sh_mid = (kps[LS, :2] + kps[RS, :2]) / 2   # shoulder midpoint [y, x]
    hip_mid = (kps[LH, :2] + kps[RH, :2]) / 2  # hip midpoint [y, x]
    delta = hip_mid - sh_mid                     # [dy, dx]
    torso_angle = math.degrees(math.atan2(abs(delta[1]), abs(delta[0]) + 1e-6))

    # Head-to-feet angle
    head_mask = confs[HEAD_IDX] > thresh
    feet_mask = confs[FEET_IDX] > thresh
    head_feet_angle = None

    if head_mask.any() and feet_mask.any():
        head_avg = kps[HEAD_IDX[head_mask], :2].mean(axis=0)  # [y, x]
        feet_avg = kps[FEET_IDX[feet_mask], :2].mean(axis=0)
        hf_d = head_avg - feet_avg
        head_feet_angle = math.degrees(math.atan2(abs(hf_d[1]), abs(hf_d[0]) + 1e-6))

    # Bounding box from all visible keypoints 
    visible = confs >= thresh
    kp_count = visible.sum()
    if kp_count < 5:
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
        'center_y': float(center_y),
        'normalized_cy': float(norm_cy),
        'bbox_height': float(bbox_h),
        'bbox_width': float(bbox_w),
        'ankle_visible': ankle_visible,
        'keypoint_count': int(kp_count),
        'shoulder_conf': float(min(confs[LS], confs[RS])),
        'hip_conf': float(min(confs[LH], confs[RH])),
        'bbox': (float(x_min), float(y_min), float(x_max), float(y_max)),
    }


def is_upright(f, torso_t=35, ratio_t=1.0):
    return f['torso_angle'] < torso_t and f['aspect_ratio'] < ratio_t


def is_fallen(f, torso_t=50, ratio_t=1.0, hf_t=55):
    if f['torso_angle'] > torso_t and f['aspect_ratio'] > ratio_t:
        return True
    if f.get('head_feet_angle') is not None and f['head_feet_angle'] > hf_t:
        return True
    if f['aspect_ratio'] > 1.8:
        return True
    return False


# Fall state tracker
class FallTracker:
    def __init__(self, confirm_time=1.5, cooldown=5.0):
        self.state = 'unknown'
        self.last_upright_t = time.time()
        self.fall_start_t = 0.0
        self.last_alert_t = 0.0
        self.cy_history = deque(maxlen=30)
        self.last_features = None
        self.confirm_time = confirm_time
        self.cooldown = cooldown
        self.frames_missing = 0
        self._first_detection = True

        # Debug histories
        self.torso_hist = deque(maxlen=90)
        self.ratio_hist = deque(maxlen=90)
        self.hf_hist = deque(maxlen=90)

    def update(self, feat, now, torso_t, ratio_t, hf_t, transition_t=1.5, record_debug=False):
        if feat is None:
            self.frames_missing += 1
            if self.frames_missing > 15 and self.state == 'potential_fall':
                self.state = 'unknown'
            return None

        self.frames_missing = 0
        self.last_features = feat

        if self._first_detection:
            self.last_upright_t = now
            self._first_detection = False

        up = is_upright(feat)
        fell = is_fallen(feat, torso_t, ratio_t, hf_t)

        if record_debug:
            self.torso_hist.append(feat['torso_angle'])
            self.ratio_hist.append(feat['aspect_ratio'])
            if feat.get('head_feet_angle') is not None:
                self.hf_hist.append(feat['head_feet_angle'])

        self.cy_history.append(feat['normalized_cy'])

        if up:
            self.last_upright_t = now

        event = None

        if self.state == 'unknown':
            if (now - self.last_upright_t) < transition_t and fell:
                self.state = 'potential_fall'
                self.fall_start_t = now

        elif self.state == 'potential_fall':
            if fell:
                if (now - self.fall_start_t) > self.confirm_time:
                    if (now - self.last_alert_t) > self.cooldown:
                        self.last_alert_t = now
                        event = 'fall'
                    self.state = 'fallen'
            elif up:
                self.state = 'unknown'

        elif self.state == 'fallen':
            if up:
                self.state = 'unknown'
                self.cy_history.clear()
                event = 'recovered'

        return event


# Drawing 
def draw_skeleton(frame, kps, thresh=0.3):
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


def draw_debug(frame, tracker, thresholds):
    """Render debug info into a separate side panel and return it."""
    h = frame.shape[0]
    pw = 260
    panel = np.zeros((h, pw + 10, 3), dtype=np.uint8)

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
        return panel

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
    fell = is_fallen(feat, tt, rt, ht)
    cv2.putText(panel, f"upright={up}  fallen={fell}", (x, y), F, 0.4,
                (0,0,255) if fell else (0,255,0) if up else W, 1)
    y += 22

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
        # Data line — sample every other point for speed
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

    return panel


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
    parser.add_argument("--model", default="movenet_lightning.tflite")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=1, help="Run inference every N frames (1=every frame, 2=every other, etc)")
    parser.add_argument("--torso-thresh", type=float, default=50)
    parser.add_argument("--ratio-thresh", type=float, default=1.0)
    parser.add_argument("--hf-thresh", type=float, default=55)
    parser.add_argument("--confirm-time", type=float, default=1.5)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    thresholds = {'torso': args.torso_thresh, 'ratio': args.ratio_thresh, 'hf': args.hf_thresh}

    print("=" * 50)
    print("  Fall Detection System ")
    print("=" * 50)
    model = PoseModel(args.model)
    print(f"  Camera: {args.camera} | Skip: every {args.skip_frames} frame(s)")
    print(f"  Thresholds: torso>{args.torso_thresh} ratio>{args.ratio_thresh} hf>{args.hf_thresh}")
    print(f"  Debug: {'ON' if args.debug else 'OFF (press d to toggle)'}")
    print(f"  Press 'q' to quit, 'd' to toggle debug\n")

    cam = CameraStream(args.camera)
    if not cam.is_opened():
        raise RuntimeError("Cannot open camera")

    # Wait for first frame
    time.sleep(0.5)

    tracker = FallTracker(confirm_time=args.confirm_time)
    show_debug = args.debug

    fps_time = time.time()
    fps_count = 0
    fps_val = 0
    inf_ms = 0
    frame_num = 0

    # Cache last inference results for skip-frame mode
    last_kps = None
    last_feat = None

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        frame_num += 1

        # Run inference every Nth frame
        run_now = (frame_num % args.skip_frames == 0)

        if run_now:
            t0 = time.time()
            last_kps = model.run(frame)
            inf_ms = (time.time() - t0) * 1000

            last_feat = find_features(last_kps, args.score_thresh)

            ev = tracker.update(last_feat, now, args.torso_thresh,
                                args.ratio_thresh, args.hf_thresh,
                                record_debug=show_debug)
            if ev == 'fall':
                print(f"[{time.strftime('%H:%M:%S')}] FALL DETECTED")
            elif ev == 'recovered':
                print(f"[{time.strftime('%H:%M:%S')}] Person recovered")

        # Headless mode 
        if args.headless:
            if run_now and frame_num % 30 == 0:
                # Print periodic status
                if last_feat:
                    hf = last_feat.get('head_feet_angle')
                    hs = f"{hf:.0f}" if hf else "?"
                    print(f"[{time.strftime('%H:%M:%S')}] {tracker.state} | "
                          f"torso={last_feat['torso_angle']:.0f} hf={hs} "
                          f"ratio={last_feat['aspect_ratio']:.1f} | {inf_ms:.0f}ms")
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
        cv2.putText(frame, f"FPS:{fps_val:.0f} inf:{inf_ms:.0f}ms skip:{args.skip_frames}",
                    (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

        # Debug panel
        if show_debug:
            debug_panel = draw_debug(frame, tracker, thresholds)
            display = np.hstack([frame, debug_panel])
        else:
            display = frame

        cv2.imshow('Fall Detection', display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"Debug: {'ON' if show_debug else 'OFF'}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()