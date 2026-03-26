"""
Peripheral controller for the fall detection system.

Connects to Pose_Estimation.py via Unix domain socket IPC and drives:
  - Buzzer (GPIO 17 active HIGH)
  - RGB LED (GPIO 22/24/23 active LOW)
  - Voice commands (Vosk offline)
  - Audio responses (pygame.mixer)

Usage:
  python peripheral_controller.py
  python peripheral_controller.py --vosk-model ./vosk-model-small-en-us-0.15
  python peripheral_controller.py --buzzer-timeout 60 --mic-device 1
"""

import time
import os
import sys
import json
import signal
import socket
import threading
import argparse
import subprocess

from gpiozero import RGBLED, PWMOutputDevice


# IPC Client
class IPCClient:
    """Connect to Pose_Estimation IPC server"""

    def __init__(self, sock_path, on_event_cb):
        self.sock_path = sock_path
        self.on_event = on_event_cb
        self._sock = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        self._running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        logged_waiting = False
        while self._running:
            try:
                self._connect()
                logged_waiting = False
                self._recv_loop()
            except Exception as e:
                if not logged_waiting:
                    print(f"[IPC] Waiting for Pose_Estimation ({e})")
                    logged_waiting = True
            time.sleep(2.0)

    def _connect(self):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(self.sock_path)
        self._sock.settimeout(5.0)
        print(f"[IPC] Connected to {self.sock_path}")

    def _recv_loop(self):
        buf = b''
        while self._running:
            try:
                data = self._sock.recv(4096)
                if not data:
                    raise ConnectionError("Server disconnected")
                buf += data
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    try:
                        msg = json.loads(line.decode())
                        self.on_event(msg)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
            except socket.timeout:
                continue

    def send_command(self, cmd):
        with self._lock:
            if self._sock:
                try:
                    self._sock.sendall(
                        (json.dumps({'type': 'command', 'cmd': cmd}) + '\n').encode())
                except Exception:
                    pass

    def stop(self):
        self._running = False
        try:
            self._sock.close()
        except Exception:
            pass


# Buzzer
class BuzzerController:
    """Control the buzzer via TIP120G transistor (active HIGH)."""

    def __init__(self, pin, timeout=30.0, frequency=2800):
        self.buzzer = PWMOutputDevice(pin, frequency=frequency)
        self.timeout = timeout
        self._active = False
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._beep_thread = None

    def activate(self):
        with self._lock:
            self._active = True
            self._start_time = time.time()
            if self._beep_thread is None or not self._beep_thread.is_alive():
                self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
                self._beep_thread.start()
        print("[BUZZER] Activated")

    def _beep_loop(self):
        """Pulsing beep pattern: 0.3s on, 0.2s off — louder and more urgent."""
        while self._active:
            self.buzzer.value = 1.0
            time.sleep(0.3)
            self.buzzer.value = 0.0
            time.sleep(0.2)

    def deactivate(self):
        with self._lock:
            self._active = False
            self.buzzer.value = 0.0
        print("[BUZZER] Deactivated")

    def check_timeout(self):
        with self._lock:
            if self._active and (time.time() - self._start_time) > self.timeout:
                self._active = False
                self.buzzer.value = 0.0
                print("[BUZZER] Auto-timeout")
                return True
        return False

    @property
    def is_active(self):
        with self._lock:
            return self._active

    def cleanup(self):
        self._active = False
        self.buzzer.value = 0.0
        self.buzzer.close()


# RGB LED (active LOW)
class LEDController:
    """RGB LED state management.

    Colors:
        green  = system running, no fall
        yellow = potential fall
        red    = fall detected
        blue   = listening / processing voice
        off    = shutdown
    """

    COLORS = {
        'ok':             (0, 1, 0),       # green
        'potential_fall': (1, 0.6, 0),     # yellow/amber
        'fallen':         (1, 0, 0),       # red
        'listening':      (0, 0, 1),       # blue
        'off':            (0, 0, 0),
    }

    def __init__(self, r_pin, g_pin, b_pin):
        self.led = RGBLED(r_pin, g_pin, b_pin, active_high=False)
        self._state = 'off'

    def set_state(self, state_name):
        color = self.COLORS.get(state_name, self.COLORS['off'])
        self.led.color = color
        self._state = state_name

    def cleanup(self):
        self.led.off()
        self.led.close()


# Voice Listener (Vosk)
class VoiceListener:
    """Continuous offline speech recognition using Vosk.

    Runs in a background thread. Matches recognized text against known
    command phrases and calls on_command(cmd_name).
    """

    COMMANDS = {
        'stop buzzer':  'stop_buzzer',
        'stop alarm':   'stop_buzzer',
        'stop':         'stop_buzzer',
        'reset fall':   'reset_fall',
        'reset':        'reset_fall',
        'send alert':   'send_alert',
        'call help':    'send_alert',
        'help':         'send_alert',
    }

    def __init__(self, model_path, on_command_cb, device=None, sample_rate=16000):
        self.on_command = on_command_cb
        self.device = device
        self.sample_rate = sample_rate      # Vosk target rate
        self._hw_rate = None                # actual
        self._running = False

        # Lazy import --> only fail if actually used
        try:
            from vosk import Model, KaldiRecognizer
            self.model = Model(model_path)
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self._available = True
        except Exception as e:
            print(f"[VOICE] Vosk init failed: {e}")
            print("[VOICE] Voice commands disabled. Install vosk and download a model.")
            self._available = False

    def start(self):
        if not self._available:
            return
        self._running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            print("[VOICE] sounddevice not installed, voice disabled")
            return

        # Detect hardware sample rate 
        hw_rate = self.sample_rate
        if self.device is not None:
            try:
                info = sd.query_devices(self.device, 'input')
                hw_rate = int(info['default_samplerate'])
            except Exception:
                pass

        need_resample = (hw_rate != self.sample_rate)
        if need_resample:
            ratio = hw_rate // self.sample_rate  # 48000/16000 = 3
            print(f"[VOICE] Hardware rate {hw_rate}Hz, downsampling {ratio}x to {self.sample_rate}Hz")

        block_size = hw_rate // 2  # 0.5s of audio at hardware rate
        try:
            with sd.RawInputStream(
                samplerate=hw_rate,
                blocksize=block_size,
                device=self.device,
                dtype='int16',
                channels=1,
            ) as stream:
                print(f"[VOICE] Listening (device={self.device})")
                while self._running:
                    data, overflowed = stream.read(block_size)
                    if need_resample:
                        samples = np.frombuffer(bytes(data), dtype=np.int16)
                        samples = samples[::ratio]
                        audio_bytes = samples.tobytes()
                    else:
                        audio_bytes = bytes(data)
                    if self.recognizer.AcceptWaveform(audio_bytes):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip().lower()
                        if text:
                            print(f"[VOICE] Heard: '{text}'")
                            self._match_command(text)
        except Exception as e:
            print(f"[VOICE] Audio stream error: {e}")

    def _match_command(self, text):
        for phrase, cmd in self.COMMANDS.items():
            if phrase in text:
                print(f"[VOICE] Command: {cmd}")
                self.on_command(cmd)
                return

    def stop(self):
        self._running = False


# Audio Player
class AudioPlayer:
    """Play pre-recorded WAV responses via pygame.mixer (fallback: aplay)."""

    RESPONSES = {
        'fall_detected':  'fall_detected.wav',
        'buzzer_stopped': 'buzzer_stopped.wav',
        'fall_reset':     'fall_reset.wav',
        'alert_sent':     'alert_sent.wav',
        'system_ready':   'system_ready.wav',
    }

    def __init__(self, audio_dir='audio/'):
        self.audio_dir = audio_dir
        self._use_pygame = False
        try:
            import pygame
            pygame.mixer.init()
            self._use_pygame = True
            print("[AUDIO] Using pygame.mixer")
        except Exception:
            print("[AUDIO] pygame.mixer unavailable falling back to aplay")

    def play(self, response_key):
        filename = self.RESPONSES.get(response_key)
        if not filename:
            return
        filepath = os.path.join(self.audio_dir, filename)
        if not os.path.exists(filepath):
            print(f"[AUDIO] Missing: {filepath}")
            return

        if self._use_pygame:
            import pygame
            try:
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
            except Exception as e:
                print(f"[AUDIO] Playback error: {e}")
        else:
            subprocess.Popen(
                ['aplay', '-D', 'plughw:0,0', filepath],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def cleanup(self):
        if self._use_pygame:
            import pygame
            pygame.mixer.quit()


# Peripheral Manager
class PeripheralManager:
    """Central coordinator: receives IPC events and voice commands,
    drives buzzer, LED, and audio responses."""

    def __init__(self, args):
        self.buzzer = BuzzerController(args.buzzer_pin, timeout=args.buzzer_timeout)
        self.led = LEDController(args.led_r, args.led_g, args.led_b)
        self.audio = AudioPlayer(args.audio_dir)
        self.ipc = IPCClient(args.ipc_sock, on_event_cb=self._on_ipc_event)
        self.voice = VoiceListener(
            args.vosk_model,
            on_command_cb=self._on_voice_command,
            device=args.mic_device,
        )
        self._current_state = 'unknown'
        self._lock = threading.Lock()

    def start(self):
        self.led.set_state('ok')
        self.audio.play('system_ready')
        self.ipc.start()
        self.voice.start()
        print("[MANAGER] All peripherals started")

    def _on_ipc_event(self, msg):
        msg_type = msg.get('type')

        if msg_type == 'event':
            event = msg.get('event')
            if event == 'fall_detected':
                self._handle_fall()
            elif event == 'recovered':
                self._handle_recovered()

        elif msg_type == 'status':
            state = msg.get('state', 'unknown')
            with self._lock:
                if self._current_state != 'fallen':
                    if state == 'unknown':
                        self.led.set_state('ok')
                    elif state == 'potential_fall':
                        self.led.set_state('potential_fall')
                        self._current_state = 'potential_fall'

    def _handle_fall(self):
        with self._lock:
            self._current_state = 'fallen'
        self.led.set_state('fallen')
        self.buzzer.activate()
        self.audio.play('fall_detected')
        print("[MANAGER] FALL -- buzzer on, LED red")

    def _handle_recovered(self):
        with self._lock:
            self._current_state = 'unknown'
        self.buzzer.deactivate()
        self.led.set_state('ok')
        print("[MANAGER] Recovered -- buzzer off, LED green")

    def _on_voice_command(self, cmd):
        if cmd == 'stop_buzzer':
            self.buzzer.deactivate()
            self.audio.play('buzzer_stopped')
            self.ipc.send_command('stop_buzzer')

        elif cmd == 'reset_fall':
            self.buzzer.deactivate()
            self.led.set_state('ok')
            with self._lock:
                self._current_state = 'unknown'
            self.audio.play('fall_reset')
            self.ipc.send_command('reset_fall')

        elif cmd == 'send_alert':
            self.audio.play('alert_sent')
            self.ipc.send_command('send_alert')
            # trigger SMS/email/notification here

    def run_forever(self):
        try:
            while True:
                if self.buzzer.check_timeout():
                    self.audio.play('buzzer_stopped')
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[MANAGER] Shutting down...")

    def cleanup(self):
        self.voice.stop()
        self.ipc.stop()
        self.buzzer.cleanup()
        self.led.cleanup()
        self.audio.cleanup()


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Peripheral Controller for Fall Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python peripheral_controller.py
  python peripheral_controller.py --vosk-model ./vosk-model-small-en-us-0.15
  python peripheral_controller.py --buzzer-timeout 60 --mic-device 1
        """)
    # IPC
    parser.add_argument("--ipc-sock", default="/tmp/falldetect.sock",
                        help="Unix socket path (must match Pose_Estimation.py --ipc-sock)")
    # GPIO (defaults match the HAT)
    parser.add_argument("--buzzer-pin", type=int, default=17,
                        help="BCM pin for buzzer transistor base (default: 17)")
    parser.add_argument("--led-r", type=int, default=22,
                        help="BCM pin for red LED (default: 22)")
    parser.add_argument("--led-g", type=int, default=24,
                        help="BCM pin for green LED (default: 24)")
    parser.add_argument("--led-b", type=int, default=23,
                        help="BCM pin for blue LED (default: 23)")
    # Vosk
    parser.add_argument("--vosk-model", default="vosk-model-small-en-us-0.15",
                        help="Path to Vosk model directory")
    parser.add_argument("--mic-device", type=int, default=None,
                        help="ALSA device index for I2S mic (None = system default)")
    # Audio
    parser.add_argument("--audio-dir", default="audio/",
                        help="Directory containing response WAV files")
    # Buzzer
    parser.add_argument("--buzzer-timeout", type=float, default=30.0,
                        help="Auto-stop buzzer after N seconds (default: 30)")
    args = parser.parse_args()

    manager = PeripheralManager(args)

    def shutdown(signum, frame):
        print("\n[MANAGER] Signal received --> cleaning")
        manager.cleanup()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    manager.start()
    manager.run_forever()
    manager.cleanup()


if __name__ == '__main__':
    main()
