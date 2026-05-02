# =============================================================================
# kafka_streamer.py — High-Volume Kafka Producer for EV Battery Telemetry
#
# Simulates millions of records from a fleet of EV sensors being streamed
# to a Kafka topic in real time, row by row from our battery CSV dataset.
#
# SETUP INSTRUCTIONS (Local, No Docker):
# ----------------------------------------
# Step 1: Download Kafka
#   - Go to: https://kafka.apache.org/downloads
#   - Download: kafka_2.13-3.7.0.tgz
#   - Extract to: C:\kafka (Windows) or ~/kafka (Mac/Linux)
#
# Step 2: Start ZooKeeper (Kafka's dependency)
#   Windows:
#     cd C:\kafka
#     .\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
#   Mac/Linux:
#     cd ~/kafka
#     ./bin/zookeeper-server-start.sh ./config/zookeeper.properties
#
# Step 3: Start Kafka Broker (new terminal)
#   Windows:
#     cd C:\kafka
#     .\bin\windows\kafka-server-start.bat .\config\server.properties
#   Mac/Linux:
#     cd ~/kafka
#     ./bin/kafka-server-start.sh ./config/server.properties
#
# Step 4: Create the topic (new terminal)
#   Windows:
#     cd C:\kafka
#     .\bin\windows\kafka-topics.bat --create --topic ev_battery_telemetry
#       --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
#   Mac/Linux:
#     ~/kafka/bin/kafka-topics.sh --create --topic ev_battery_telemetry
#       --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
#
# Step 5: Install Python dependency
#   pip install confluent-kafka
#
# Step 6: Run this producer
#   python kafka_streamer.py
#   python kafka_streamer.py --delay 0.01   # faster streaming
#   python kafka_streamer.py --delay 0      # maximum speed (millions/hour)
# =============================================================================

import json
import time
import argparse
import signal
import sys
import os
from datetime import datetime
from confluent_kafka import Producer, KafkaException
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BROKER    = "localhost:9092"
TOPIC_NAME      = "ev_battery_telemetry"
CSV_PATH        = "Battery_Data_Cleaned.csv"
DEFAULT_DELAY   = 0.05    # seconds between messages (0.05s = 20 msgs/sec)

# Kafka producer config — tuned for high throughput
PRODUCER_CONFIG = {
    "bootstrap.servers"    : KAFKA_BROKER,
    "acks"                 : "1",           # wait for leader ack only (faster than "all")
    "linger.ms"            : "5",           # batch messages for 5ms before sending
    "batch.size"           : "65536",       # 64KB batch size
    "compression.type"     : "snappy",      # compress batches (reduces network load)
    "retries"              : "3",
    "retry.backoff.ms"     : "100",
}

# ---------------------------------------------------------------------------
# Delivery callback — called for every message Kafka confirms or fails
# ---------------------------------------------------------------------------

messages_sent    = 0
messages_failed  = 0
total_loops      = 0

def delivery_callback(err, msg):
    """
    Called by Kafka producer for every message after it is acknowledged.
    err is None on success, contains error details on failure.
    """
    global messages_sent, messages_failed

    if err:
        messages_failed += 1
        print(f"  [FAILED] Partition {msg.partition()} | Error: {err}")
    else:
        messages_sent += 1


# ---------------------------------------------------------------------------
# Signal handler — graceful shutdown on Ctrl+C
# ---------------------------------------------------------------------------

running = True

def handle_shutdown(signum, frame):
    global running
    print(f"\n\n  Shutting down gracefully...")
    print(f"  Total messages sent   : {messages_sent:,}")
    print(f"  Total messages failed : {messages_failed:,}")
    print(f"  Dataset loops         : {total_loops}")
    running = False


signal.signal(signal.SIGINT,  handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)


# ---------------------------------------------------------------------------
# Main producer function
# ---------------------------------------------------------------------------

def stream_battery_data(csv_path: str, delay: float):
    """
    Infinitely loops over the CSV file and publishes each row to Kafka.
    Simulates a real-world fleet of EVs continuously sending sensor data.

    Each message contains:
    - battery_id, test_id, Capacity, Re, Rct, ambient_temperature
    - Plus a real timestamp and loop_count to track simulation progress

    With delay=0.05s this produces ~20 messages/second = ~1.7M messages/day
    With delay=0.0  this produces maximum throughput (CPU limited)
    """
    global total_loops, running

    # Load CSV once — keep in memory for speed
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"  Dataset loaded: {len(df):,} rows across "
          f"{df['battery_id'].nunique()} batteries")

    # Keep only the columns we need for the model
    required_cols = ["battery_id", "test_id", "Capacity",
                     "Re", "Rct", "ambient_temperature"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        sys.exit(1)

    df = df[required_cols].copy()
    total_rows = len(df)

    # Initialize Kafka producer
    try:
        producer = Producer(PRODUCER_CONFIG)
        print(f"  Kafka producer connected to: {KAFKA_BROKER}")
        print(f"  Publishing to topic        : {TOPIC_NAME}")
        print(f"  Delay between messages     : {delay}s")
        print(f"  Estimated throughput       : {1/delay:.0f} msg/s" if delay > 0 else "  Throughput: Maximum speed")
        print("\n  Starting infinite stream — Ctrl+C to stop\n")
        print("=" * 60)
    except KafkaException as e:
        print(f"ERROR: Could not connect to Kafka broker: {e}")
        print("Make sure ZooKeeper and Kafka are running (see setup instructions)")
        sys.exit(1)

    loop_count     = 0
    last_report_ts = time.time()

    while running:
        loop_count   += 1
        total_loops   = loop_count

        for idx, row in df.iterrows():
            if not running:
                break

            # Build the message payload as JSON
            message = {
                "battery_id"          : int(row["battery_id"]),
                "test_id"             : int(row["test_id"]),
                "Capacity"            : float(row["Capacity"]),
                "Re"                  : float(row["Re"]),
                "Rct"                 : float(row["Rct"]),
                "ambient_temperature" : float(row["ambient_temperature"]),
                "timestamp"           : datetime.utcnow().isoformat(),
                "simulation_loop"     : loop_count,
                "record_index"        : int(idx),
            }

            # Use battery_id as the Kafka partition key
            # This ensures all data for one battery goes to the same partition
            # which is critical for the sliding window logic in the consumer
            partition_key = str(row["battery_id"]).encode("utf-8")
            message_value = json.dumps(message).encode("utf-8")

            # Produce message — non-blocking, delivery_callback handles acks
            try:
                producer.produce(
                    topic    = TOPIC_NAME,
                    key      = partition_key,
                    value    = message_value,
                    callback = delivery_callback
                )
            except BufferError:
                # Producer queue is full — poll to free space then retry
                producer.poll(0.1)
                producer.produce(
                    topic    = TOPIC_NAME,
                    key      = partition_key,
                    value    = message_value,
                    callback = delivery_callback
                )

            # Poll to trigger delivery callbacks — do this every 100 messages
            if idx % 100 == 0:
                producer.poll(0)

            # Print progress every 10 seconds
            now = time.time()
            if now - last_report_ts >= 10:
                rate = messages_sent / max(now - (last_report_ts - 10), 1)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
                      f"Loop {loop_count:>4} | "
                      f"Row {idx:>5}/{total_rows} | "
                      f"Sent: {messages_sent:>8,} | "
                      f"Failed: {messages_failed:>4,} | "
                      f"~{messages_sent/(now - start_time):.0f} msg/s")
                last_report_ts = now

            if delay > 0:
                time.sleep(delay)

        print(f"\n  Loop {loop_count} complete — "
              f"{total_rows:,} records published. "
              f"Starting loop {loop_count + 1}...\n")

    # Flush remaining messages before exit
    print(f"\n  Flushing remaining messages...")
    producer.flush(timeout=10)
    print(f"  Producer shutdown complete.")
    print(f"  Grand total sent: {messages_sent:,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EV Battery Kafka Producer — streams CSV data infinitely"
    )
    parser.add_argument(
        "--csv",
        default=CSV_PATH,
        help=f"Path to battery CSV file (default: {CSV_PATH})"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay in seconds between messages (default: {DEFAULT_DELAY}). "
             f"Use 0 for maximum speed."
    )
    parser.add_argument(
        "--broker",
        default=KAFKA_BROKER,
        help=f"Kafka broker address (default: {KAFKA_BROKER})"
    )

    args = argparse.ArgumentParser().parse_args([])
    args = parser.parse_args()

    KAFKA_BROKER = args.broker
    PRODUCER_CONFIG["bootstrap.servers"] = KAFKA_BROKER

    print("=" * 60)
    print("  EV Battery Telemetry — Kafka Producer")
    print("=" * 60)

    start_time = time.time()
    stream_battery_data(args.csv, args.delay)