import websocket
import json
import _thread
import rel

NGROK_URL  = 'unabsorbing-perla-subsequently.ngrok-free.dev'
UPLOAD_URL = f'https://{NGROK_URL}/uploads/imu_data.csv'
WS_URL     = f'wss://{NGROK_URL}/ws?id=phone'


def on_message(ws, message):
    try:
        jso = json.loads(message)
    except json.JSONDecodeError:
        print(f"[PHONE] Bad JSON: {message}")
        return

    if 'error' in jso:
        print(f"[PHONE] Server error: {jso['error']}")
        return

    data = jso.get('data')
    if not data or not isinstance(data, list):
        return

    d = data[0]

    if d == 'fall':
        print("[PHONE] FALL ALERT received")

    elif d == 'no_fall':
        print("[PHONE] No fall")

    elif d == 'pic':
        url = data[1] if len(data) > 1 else '(no url)'
        print(f"[PHONE] Snapshot received: {url}")

    elif d == 'cancel':
        print("[PHONE] Alert cancelled")

    else:
        print(f"[PHONE] Unknown message: {data}")


def on_error(ws, error):
    print(f"[PHONE] Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("[PHONE] Connection closed")


def on_open(ws):
    print("[PHONE] Connected")
    ws.send(json.dumps({
        'target': 'hub',
        'status': 'upload',
        'data':   UPLOAD_URL,
    }))


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(dispatcher=rel, reconnect=5)
    rel.signal(2, rel.abort)
    rel.dispatch()