#phone
#signal to server of upload
#wait for result
#receive result
#close

from websocket import create_connection
import json

#end = "ws://localhost:5000/ws?id=phone"
end = "wss://1327-68-148-232-205.ngrok-free.app/ws?id=phone"
ws = create_connection(end)
#print(ws.recv())
print("Sending Upload Notice")
ws.send(json.dumps({'target': 'hub', 'status': 'upload','data':'https://1327-68-148-232-205.ngrok-free.app/uploads/imu_data.csv'}))
print("Sent")
print("Receiving result")
result =  ws.recv()
print("Received '%s'" % result)
ws.close()