# websocket_consumer.py

import redis
import asyncio
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque
from starlette.websockets import WebSocketDisconnect

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

WINDOW_SIZE = 50
data_buffer = deque(maxlen=WINDOW_SIZE + 1)

async def read_stream_and_predict(websocket, model):
    last_id = '0'
    data_buffer.clear()

    try:
        await websocket.send_json({
            "type"   : "connected",
            "message": "Live battery stream started. Waiting for data..."
        })
    except Exception:
        return

    while True:
        try:
            # Client disconnect check karo
            # Non-blocking receive attempt
            try:
                from starlette.websockets import WebSocketState
                if websocket.client_state == WebSocketState.DISCONNECTED:
                    break
            except Exception:
                pass

            # Redis se data lo — thread executor mein taake async block na ho
            loop = asyncio.get_event_loop()
            messages = await loop.run_in_executor(
                None,
                lambda: r.xread({'battery:stream': last_id}, count=1, block=200)
            )

            if not messages:
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
                continue

            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    last_id = msg_id

                    cycle_data = {
                        'Capacity'           : float(msg_data['capacity']),
                        'Re'                 : float(msg_data['re']),
                        'Rct'                : float(msg_data['rct']),
                        'ambient_temperature': float(msg_data['ambient_temperature']),
                    }

                    data_buffer.append(cycle_data)

                    current_cycle    = int(msg_data['cycle'])
                    current_capacity = float(msg_data['capacity'])
                    prediction       = None

                    if len(data_buffer) >= WINDOW_SIZE:
                        window_data   = list(data_buffer)[-WINDOW_SIZE:]
                        window_arr    = np.array([
                            [d['Capacity'], d['Re'],
                             d['Rct'], d['ambient_temperature']]
                            for d in window_data
                        ], dtype=np.float32)

                        scaler        = MinMaxScaler()
                        window_scaled = scaler.fit_transform(window_arr)
                        X             = window_scaled.reshape(1, WINDOW_SIZE, 4)

                        pred_scaled   = model.predict(X, verbose=0).flatten()[0]

                        dummy         = np.zeros((1, 4))
                        dummy[0, 0]   = pred_scaled
                        prediction    = float(scaler.inverse_transform(dummy)[0, 0])

                    payload = {
                        "type"       : "cycle_data",
                        "cycle"      : current_cycle,
                        "capacity"   : round(current_capacity, 4),
                        "prediction" : round(prediction, 4) if prediction else None,
                        "battery_id" : int(msg_data['battery_id']),
                        "buffer_size": len(data_buffer)
                    }

                    try:
                        await websocket.send_json(payload)
                    except Exception:
                        return

        except WebSocketDisconnect:
            break
        except Exception as e:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
            break