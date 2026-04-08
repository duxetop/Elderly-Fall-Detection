import os
from flask import Flask, flash, request, redirect, url_for,send_from_directory
from werkzeug.utils import secure_filename
from flask_sock import Sock
import json

UPLOAD_FOLDER = './up'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
sock = Sock(app)

clients = {}
@sock.route('/ws')
def handle_socket(ws):
    device_id = request.args.get('id')
    if not device_id:
        return # Reject if no ID provided

    # Register the device
    clients[device_id] = ws
    print(f"Device {device_id} connected.")

    try:
        while True:
            data = ws.receive()
            message = json.loads(data)

            # Expecting message format: {"target": "device_b", "status": "online"}
            target_id = message.get("target")
            payload = message.get("data")

            if target_id in clients:
                target_ws = clients[target_id]
                try:
                    print("sending data to "+str(target_id)+" from "+str(device_id))
                    target_ws.send(json.dumps({
                        "from": device_id,
                        "data": payload
                    }))
                except Exception:
                    print("removing broken target")
                    # Handle broken connection to target
                    del clients[target_id]
            else:
                ws.send(json.dumps({"error": f"Device {target_id} not found"}))

    except Exception as e:
        print(f"Connection lost for {device_id}: {e}")
    finally:
        # Clean up on disconnect
        if device_id in clients:
            del clients[device_id]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/uploads/<name>')
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
