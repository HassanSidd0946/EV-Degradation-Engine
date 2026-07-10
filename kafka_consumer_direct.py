# kafka_consumer_direct.py
# PySpark ke bina seedha Kafka se data padhta hai
# aur TCN model se predictions karta hai

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from confluent_kafka import Consumer, KafkaError
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

# ── Config ────────────────────────────────────────────────────
KAFKA_BROKER  = "localhost:9092"
TOPIC_NAME    = "ev_battery_telemetry"
MODEL_PATH    = "best_tcn.keras"
WINDOW_SIZE   = 50
FEATURES      = ["Capacity", "Re", "Rct", "ambient_temperature"]
GROUP_ID      = "ev_soh_consumer_group"

# ── Load Model ────────────────────────────────────────────────
print("=" * 60)
print("  EV Battery SOH — Direct Kafka Consumer + TCN Inference")
print("=" * 60)
print(f"  Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"  Model loaded successfully")
print(f"  Kafka broker : {KAFKA_BROKER}")
print(f"  Topic        : {TOPIC_NAME}")
print(f"  Window size  : {WINDOW_SIZE} cycles")
print("=" * 60)

# ── Kafka Consumer Config ─────────────────────────────────────
consumer_config = {
    "bootstrap.servers"               : KAFKA_BROKER,
    "group.id"                        : GROUP_ID,
    "auto.offset.reset"               : "latest",
    "enable.auto.commit"              : True,
    "auto.commit.interval.ms"         : 1000,
    "session.timeout.ms"              : 30000,
    "max.poll.interval.ms"            : 300000,
}

consumer = Consumer(consumer_config)
consumer.subscribe([TOPIC_NAME])
print(f"\n  Subscribed to topic: {TOPIC_NAME}")
print(f"  Waiting for messages... (Ctrl+C to stop)\n")

# ── Per-battery State ─────────────────────────────────────────
# Each battery gets its own deque of last WINDOW_SIZE cycles
battery_buffers  = defaultdict(list)
messages_received = 0
predictions_made  = 0
start_time        = time.time()

def run_inference(battery_id: int, buffer: list) -> dict:
    """Run TCN inference on a battery's cycle buffer."""
    window_df     = np.array(
        [[r["Capacity"], r["Re"], r["Rct"], r["ambient_temperature"]]
         for r in buffer[-WINDOW_SIZE:]],
        dtype=np.float32
    )
    scaler        = MinMaxScaler()
    window_scaled = scaler.fit_transform(window_df)
    X             = window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))

    pred_scaled   = model.predict(X, verbose=0).flatten()[0]
    dummy         = np.zeros((1, len(FEATURES)))
    dummy[0, 0]   = pred_scaled
    prediction_ah = float(scaler.inverse_transform(dummy)[0, 0])

    actual        = float(buffer[-1]["Capacity"])
    first         = float(buffer[0]["Capacity"])
    soh           = round((prediction_ah / first) * 100, 2) if first > 0 else 0.0

    status = "HEALTHY" if soh >= 85 else "CAUTION" if soh >= 70 else "REPLACE"

    return {
        "battery_id"   : battery_id,
        "cycle"        : buffer[-1].get("test_id", -1),
        "actual_ah"    : round(actual, 4),
        "predicted_ah" : round(prediction_ah, 4),
        "soh_percent"  : soh,
        "status"       : status,
        "buffer_size"  : len(buffer),
    }

# ── Main Consumer Loop ────────────────────────────────────────
try:
    last_report = time.time()

    while True:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"  Kafka error: {msg.error()}")
                break

        # Parse JSON message
        try:
            data = json.loads(msg.value().decode("utf-8"))
        except Exception as e:
            print(f"  Parse error: {e}")
            continue

        battery_id = data.get("battery_id")
        if not battery_id:
            continue

        messages_received += 1

        # Add to battery buffer
        battery_buffers[battery_id].append(data)

        # Keep only last 200 cycles per battery to save memory
        if len(battery_buffers[battery_id]) > 200:
            battery_buffers[battery_id] = battery_buffers[battery_id][-200:]

        # Run inference when buffer has enough cycles
        if len(battery_buffers[battery_id]) >= WINDOW_SIZE:
            try:
                result = run_inference(battery_id, battery_buffers[battery_id])
                predictions_made += 1

                print(f"  Battery {result['battery_id']:>3} | "
                      f"Cycle {result['cycle']:>4} | "
                      f"Actual: {result['actual_ah']:.4f} Ah | "
                      f"Predicted: {result['predicted_ah']:.4f} Ah | "
                      f"SOH: {result['soh_percent']:>6.1f}% | "
                      f"{result['status']}")

            except Exception as e:
                print(f"  Inference error battery {battery_id}: {e}")

        # Print throughput every 30 seconds
        now = time.time()
        if now - last_report >= 30:
            elapsed  = now - start_time
            rate     = messages_received / elapsed
            active   = len(battery_buffers)
            print(f"\n  --- Stats | "
                  f"Messages: {messages_received:,} | "
                  f"Rate: {rate:.1f} msg/s | "
                  f"Active batteries: {active} | "
                  f"Predictions: {predictions_made:,} ---\n")
            last_report = now

except KeyboardInterrupt:
    print(f"\n\n  Stopping consumer...")
    print(f"  Total messages received : {messages_received:,}")
    print(f"  Total predictions made  : {predictions_made:,}")
    elapsed = time.time() - start_time
    print(f"  Runtime                 : {elapsed:.0f}s")
    print(f"  Average rate            : {messages_received/elapsed:.1f} msg/s")

finally:
    consumer.close()
    print("  Consumer closed.")