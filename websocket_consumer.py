# websocket_consumer.py
# Redis stream se data read karke WebSocket clients ko bhejta hai

import redis
import json
import asyncio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# Redis connection
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

WINDOW_SIZE = 50
FEATURES    = ["Capacity", "Re", "Rct", "ambient_temperature"]

# Sliding window buffer — last 50 cycles store karta hai
data_buffer = deque(maxlen=WINDOW_SIZE + 1)

async def read_stream_and_predict(websocket, model):
    last_id = '0'   # Redis stream ka starting point

    await websocket.send_json({
        "type"   : "connected",
        "message": "Live battery stream started. Waiting for data..."
    })

    while True:
        try:
            # Redis stream se naya data padhon
            messages = r.xread(
                {'battery:stream': last_id},
                count=1,
                block=500    # 500ms wait karo naye data ka
            )

            if not messages:
                # Koi naya data nahi aaya — client ko heartbeat bhejo
                await websocket.send_json({"type": "heartbeat"})
                await asyncio.sleep(0.1)
                continue

            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    last_id = msg_id

                    # Data parse karo
                    cycle_data = {
                        'Capacity'           : float(msg_data['capacity']),
                        'Re'                 : float(msg_data['re']),
                        'Rct'                : float(msg_data['rct']),
                        'ambient_temperature': float(msg_data['ambient_temperature']),
                    }

                    # Buffer mein add karo
                    data_buffer.append(cycle_data)

                    current_cycle    = int(msg_data['cycle'])
                    current_capacity = float(msg_data['capacity'])

                    # 50 cycles buffer mein aa gaye toh prediction karo
                    prediction = None
                    if len(data_buffer) >= WINDOW_SIZE:

                        window_data = list(data_buffer)[-WINDOW_SIZE:]
                        window_df   = np.array([
                            [d['Capacity'], d['Re'],
                             d['Rct'], d['ambient_temperature']]
                            for d in window_data
                        ], dtype=np.float32)

                        # Scale karo
                        scaler = MinMaxScaler()
                        window_scaled = scaler.fit_transform(window_df)
                        X = window_scaled.reshape(1, WINDOW_SIZE, 4)

                        # Predict karo
                        pred_scaled = model.predict(X, verbose=0).flatten()[0]

                        # Inverse scale
                        dummy       = np.zeros((1, 4))
                        dummy[0, 0] = pred_scaled
                        prediction  = float(scaler.inverse_transform(dummy)[0, 0])

                    # WebSocket se frontend ko bhejo
                    payload = {
                        "type"       : "cycle_data",
                        "cycle"      : current_cycle,
                        "capacity"   : round(current_capacity, 4),
                        "prediction" : round(prediction, 4) if prediction else None,
                        "battery_id" : int(msg_data['battery_id']),
                        "buffer_size": len(data_buffer)
                    }

                    await websocket.send_json(payload)

        except Exception as e:
            await websocket.send_json({
                "type" : "error",
                "message": str(e)
            })
            break