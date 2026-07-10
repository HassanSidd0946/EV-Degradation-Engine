# =============================================================================
# pyspark_consumer.py — PySpark Structured Streaming + TCN Deep Learning Inference
#
# Reads from Kafka topic ev_battery_telemetry, applies sliding window logic
# grouped by battery_id, and runs TCN model inference using Pandas UDFs.
#
# SETUP INSTRUCTIONS (Local, No Docker):
# ----------------------------------------
# Step 1: Install Java (PySpark requires Java 8 or 11)
#   - Download: https://adoptium.net/
#   - Install Java 11 LTS
#   - Set JAVA_HOME environment variable:
#     Windows: setx JAVA_HOME "C:\Program Files\Eclipse Adoptium\jdk-11..."
#     Mac/Linux: export JAVA_HOME=$(/usr/libexec/java_home -v 11)
#
# Step 2: Install PySpark and dependencies
#   pip install pyspark==3.5.0 pandas numpy scikit-learn
#
# Step 3: Download Kafka-Spark connector JAR (needed for Spark to read Kafka)
#   Download this file and place it in your project folder:
#   https://repo1.maven.org/maven2/org/apache/spark/
#           spark-sql-kafka-0-10_2.12/3.5.0/
#           spark-sql-kafka-0-10_2.12-3.5.0.jar
#
#   Or let Spark auto-download it using --packages (see Step 5)
#
# Step 4: Make sure your Kafka broker is running (see kafka_streamer.py setup)
#   And the kafka_streamer.py is actively producing messages
#
# Step 5: Run this PySpark consumer
#   Basic run:
#     python pyspark_consumer.py
#
#   With auto-downloaded Kafka connector (recommended first time):
#     spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0
#       pyspark_consumer.py
#
#   With local JAR file:
#     spark-submit --jars spark-sql-kafka-0-10_2.12-3.5.0.jar
#       pyspark_consumer.py
#
# ARCHITECTURE:
#   Kafka Topic -> Spark Structured Streaming -> Window Aggregation
#   -> Pandas UDF (TCN Model) -> Console Output / Delta Table
# =============================================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from collections import deque

# Python executable ka exact path set karo
python_path = sys.executable
print(f"Using Python: {python_path}")

os.environ["HADOOP_HOME"]           = "C:\\hadoop"
os.environ["hadoop.home.dir"]       = "C:\\hadoop"
os.environ["PYSPARK_PYTHON"]        = "C:\\Program Files\\Python312\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Program Files\\Python312\\python.exe"
os.environ["SPARK_LOCAL_IP"]        = "127.0.0.1"

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PySpark imports
# ---------------------------------------------------------------------------

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField,
        IntegerType, FloatType, StringType, TimestampType, DoubleType
    )
    from pyspark.sql.functions import pandas_udf, PandasUDFType
except ImportError:
    print("ERROR: PySpark not installed. Run: pip install pyspark==3.5.0")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BROKER    = "localhost:9092"
TOPIC_NAME      = "ev_battery_telemetry"
MODEL_PATH      = "best_tcn.keras"      # path to your saved TCN model
WINDOW_SIZE     = 50                        # must match training config
FEATURES        = ["Capacity", "Re", "Rct", "ambient_temperature"]
CHECKPOINT_DIR  = "C:/tmp/ev_battery_checkpoint"
OUTPUT_MODE     = "append"                  # append, complete, or update

# ---------------------------------------------------------------------------
# Build Spark Session — optimized for local execution
# ---------------------------------------------------------------------------

def create_spark_session() -> SparkSession:
    """
    Creates a SparkSession configured for local execution with Kafka support.
    local[*] uses all available CPU cores.
    """
    spark = (
        SparkSession.builder
        .appName("EV_Battery_SOH_Streaming")
        .master("local[*]")          # use all CPU cores locally
        .config("spark.hadoop.validateOutputSpecs", "false")
        .config("spark.sql.warehouse.dir", "C:/tmp/spark-warehouse")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"
        )
        # Memory settings — adjust based on your machine
        .config("spark.driver.memory",   "4g")
        .config("spark.executor.memory", "4g")

        # Shuffle partitions — lower for local dev (default 200 is too many)
        .config("spark.sql.shuffle.partitions", "8")

        # Kafka micro-batch settings — process every 5 seconds
        .config("spark.streaming.kafka.consumer.poll.ms", "512")

        # Serializer for performance
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

        # Logging level — reduce noise
        .config("spark.ui.showConsoleProgress", "false")
        
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark


# ---------------------------------------------------------------------------
# Schema for incoming Kafka JSON messages
# ---------------------------------------------------------------------------

MESSAGE_SCHEMA = StructType([
    StructField("battery_id",          IntegerType(),   True),
    StructField("test_id",             IntegerType(),   True),
    StructField("Capacity",            FloatType(),     True),
    StructField("Re",                  FloatType(),     True),
    StructField("Rct",                 FloatType(),     True),
    StructField("ambient_temperature", FloatType(),     True),
    StructField("timestamp",           StringType(),    True),
    StructField("simulation_loop",     IntegerType(),   True),
    StructField("record_index",        IntegerType(),   True),
])


# ---------------------------------------------------------------------------
# Schema for the output of our Pandas UDF (TCN inference result)
# ---------------------------------------------------------------------------

PREDICTION_SCHEMA = StructType([
    StructField("battery_id",     IntegerType(), True),
    StructField("latest_cycle",   IntegerType(), True),
    StructField("actual_capacity",DoubleType(),  True),
    StructField("tcn_prediction", DoubleType(),  True),
    StructField("soh_percent",    DoubleType(),  True),
    StructField("mae_window",     DoubleType(),  True),
    StructField("status",         StringType(),  True),
])


# ---------------------------------------------------------------------------
# Pandas UDF — runs TCN model inference on each battery's window of data
#
# This is the core of the distributed inference pipeline.
# Spark will call this function once per battery_id group with a Pandas DataFrame
# containing all rows for that battery in the current micro-batch.
# ---------------------------------------------------------------------------

@pandas_udf(PREDICTION_SCHEMA, PandasUDFType.GROUPED_MAP)
def tcn_inference_udf(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Pandas UDF that applies sliding window + TCN model inference.

    Called by Spark once per battery_id group in each micro-batch.
    pdf: Pandas DataFrame with all rows for one battery in this batch.

    Returns a single-row Pandas DataFrame with the prediction result.
    """
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    # Sort by test_id to ensure chronological order
    pdf = pdf.sort_values("test_id").reset_index(drop=True)

    battery_id = int(pdf["battery_id"].iloc[0])

    # We need at least WINDOW_SIZE rows to make a prediction
    if len(pdf) < WINDOW_SIZE:
        return pd.DataFrame([{
            "battery_id"     : battery_id,
            "latest_cycle"   : int(pdf["test_id"].iloc[-1]) if len(pdf) > 0 else -1,
            "actual_capacity": float(pdf["Capacity"].iloc[-1]) if len(pdf) > 0 else 0.0,
            "tcn_prediction" : 0.0,
            "soh_percent"    : 0.0,
            "mae_window"     : 0.0,
            "status"         : f"BUFFERING ({len(pdf)}/{WINDOW_SIZE} cycles)"
        }])

    try:
        # Load model — TF caches this after first load in each executor
        # so subsequent calls in the same executor reuse the loaded model
        model = tf.keras.models.load_model(MODEL_PATH)

        # Take the last WINDOW_SIZE rows as the input window
        window_df = pdf[FEATURES].iloc[-WINDOW_SIZE:].values.astype(np.float32)

        # Scale the window
        scaler        = MinMaxScaler()
        window_scaled = scaler.fit_transform(window_df)
        X             = window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))

        # Run TCN inference
        pred_scaled = model.predict(X, verbose=0).flatten()[0]

        # Inverse scale back to original Ah units
        dummy         = np.zeros((1, len(FEATURES)))
        dummy[0, 0]   = pred_scaled
        prediction_ah = float(scaler.inverse_transform(dummy)[0, 0])

        # Get actual capacity of the next cycle (last row in window)
        actual_capacity = float(pdf["Capacity"].iloc[-1])
        latest_cycle    = int(pdf["test_id"].iloc[-1])

        # Calculate MAE over the window
        actual_scaled = scaler.transform(pdf[FEATURES].iloc[-WINDOW_SIZE:].values)
        actual_cap_scaled = actual_scaled[:, 0]
        all_preds = []
        for i in range(len(actual_cap_scaled)):
            window_i = window_scaled[max(0, i-WINDOW_SIZE+1):i+1]
            if len(window_i) == WINDOW_SIZE:
                X_i   = window_i.reshape(1, WINDOW_SIZE, len(FEATURES))
                p_i   = model.predict(X_i, verbose=0).flatten()[0]
                all_preds.append(p_i)

        mae_window = float(np.mean(np.abs(
            np.array(all_preds) - actual_cap_scaled[-len(all_preds):]
        ))) if all_preds else 0.0

        # Calculate SOH percentage
        first_capacity = float(pdf["Capacity"].iloc[0])
        soh_percent    = round((prediction_ah / first_capacity) * 100, 2) \
                         if first_capacity > 0 else 0.0

        # Determine health status
        if soh_percent >= 85:
            status = "HEALTHY"
        elif soh_percent >= 70:
            status = "CAUTION"
        else:
            status = "REPLACE"

        return pd.DataFrame([{
            "battery_id"     : battery_id,
            "latest_cycle"   : latest_cycle,
            "actual_capacity": round(actual_capacity, 6),
            "tcn_prediction" : round(prediction_ah, 6),
            "soh_percent"    : soh_percent,
            "mae_window"     : round(mae_window, 6),
            "status"         : status
        }])

    except Exception as e:
        # Never crash the stream — return error row instead
        return pd.DataFrame([{
            "battery_id"     : battery_id,
            "latest_cycle"   : -1,
            "actual_capacity": 0.0,
            "tcn_prediction" : 0.0,
            "soh_percent"    : 0.0,
            "mae_window"     : 0.0,
            "status"         : f"ERROR: {str(e)[:100]}"
        }])


# ---------------------------------------------------------------------------
# State management — accumulate cycles per battery across micro-batches
#
# Without this, each micro-batch would only see a few rows per battery
# and never accumulate enough for a 50-cycle window.
# ---------------------------------------------------------------------------

# Global in-memory state — maps battery_id to its accumulated DataFrame
# In production, this would be in Redis or a distributed state store
BATTERY_STATE: dict[int, pd.DataFrame] = {}

def process_micro_batch(batch_df, batch_id: int):
    """
    Called by Spark for every micro-batch of incoming Kafka messages.
    Accumulates data per battery and runs TCN inference when window is full.
    """
    if batch_df.isEmpty():
        return

    # Convert Spark DataFrame to Pandas for stateful processing
    pandas_batch = batch_df.toPandas()
    total_rows   = len(pandas_batch)

    print(f"\n{'='*60}")
    print(f"  Micro-batch {batch_id} | {total_rows} new records")

    predictions = []

    # Group by battery_id — accumulate state
    for battery_id, group in pandas_batch.groupby("battery_id"):
        battery_id = int(battery_id)

        # Append new rows to existing state for this battery
        if battery_id not in BATTERY_STATE:
            BATTERY_STATE[battery_id] = group.copy()
        else:
            BATTERY_STATE[battery_id] = pd.concat(
                [BATTERY_STATE[battery_id], group],
                ignore_index=True
            ).drop_duplicates(subset=["test_id"]).sort_values("test_id")

        accumulated = BATTERY_STATE[battery_id]

        # Only run inference if we have enough cycles
        if len(accumulated) >= WINDOW_SIZE:
            # Run TCN inference on accumulated window
            result = run_tcn_inference(battery_id, accumulated)
            if result:
                predictions.append(result)
                print(f"  Battery {battery_id:>3} | "
                      f"Cycles: {len(accumulated):>4} | "
                      f"Actual: {result['actual_capacity']:.4f} Ah | "
                      f"Predicted: {result['tcn_prediction']:.4f} Ah | "
                      f"SOH: {result['soh_percent']:.1f}% | "
                      f"{result['status']}")
        else:
            print(f"  Battery {battery_id:>3} | "
                  f"Buffering: {len(accumulated)}/{WINDOW_SIZE} cycles")

    if predictions:
        print(f"\n  Predictions this batch: {len(predictions)}")
    print(f"{'='*60}")


def run_tcn_inference(battery_id: int, pdf: pd.DataFrame) -> dict:
    """
    Runs TCN model inference on a single battery's accumulated data.
    Separated from process_micro_batch for clarity and reusability.
    """
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler

    try:
        # Use last WINDOW_SIZE rows
        window_df     = pdf[FEATURES].iloc[-WINDOW_SIZE:].values.astype(np.float32)
        scaler        = MinMaxScaler()
        window_scaled = scaler.fit_transform(window_df)
        X             = window_scaled.reshape(1, WINDOW_SIZE, len(FEATURES))

        model         = tf.keras.models.load_model(MODEL_PATH)
        pred_scaled   = model.predict(X, verbose=0).flatten()[0]

        dummy         = np.zeros((1, len(FEATURES)))
        dummy[0, 0]   = pred_scaled
        prediction_ah = float(scaler.inverse_transform(dummy)[0, 0])

        actual_capacity = float(pdf["Capacity"].iloc[-1])
        first_capacity  = float(pdf["Capacity"].iloc[0])
        soh_percent     = round((prediction_ah / first_capacity) * 100, 2) \
                          if first_capacity > 0 else 0.0

        status = ("HEALTHY" if soh_percent >= 85
                  else "CAUTION" if soh_percent >= 70
                  else "REPLACE")

        return {
            "battery_id"      : battery_id,
            "latest_cycle"    : int(pdf["test_id"].iloc[-1]),
            "actual_capacity" : round(actual_capacity, 6),
            "tcn_prediction"  : round(prediction_ah, 6),
            "soh_percent"     : soh_percent,
            "status"          : status,
        }

    except Exception as e:
        print(f"  ERROR on battery {battery_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# Main streaming pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  EV Battery SOH — PySpark Structured Streaming Consumer")
    print("=" * 60)
    print(f"  Kafka broker : {KAFKA_BROKER}")
    print(f"  Topic        : {TOPIC_NAME}")
    print(f"  Model        : {MODEL_PATH}")
    print(f"  Window size  : {WINDOW_SIZE} cycles")
    print("=" * 60)

    # Validate model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Make sure best_tcn.keras is in the same directory")
        sys.exit(1)

    spark = create_spark_session()
    print(f"\n  Spark version  : {spark.version}")
    print(f"  Spark UI       : http://localhost:4040")
    print(f"  Waiting for Kafka messages...\n")

    # Read from Kafka as a streaming DataFrame
    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKER)
        .option("subscribe",               TOPIC_NAME)
        .option("startingOffsets",         "latest")    # only new messages
        .option("maxOffsetsPerTrigger",    "1000")      # process 1000 msgs per batch
        .option("failOnDataLoss",          "false")     # don't crash if topic missing
        .load()
    )

    # Parse JSON messages from Kafka
    # Kafka delivers messages as binary (key, value) — we decode value as JSON
    parsed_stream = (
        kafka_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                MESSAGE_SCHEMA
            ).alias("data"),
            F.col("timestamp").alias("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
        .filter(F.col("battery_id").isNotNull())
        .filter(F.col("Capacity").isNotNull())
    )

    # Write stream using foreachBatch — gives us full Pandas control
    # This is where our stateful TCN inference happens
    query = (
        parsed_stream
        .writeStream
        .foreachBatch(process_micro_batch)
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", "C:/tmp/ev_battery_checkpoint")
        .start()
    )

    print(f"  Streaming started. Query ID: {query.id}")
    print(f"  Checkpoint dir : {CHECKPOINT_DIR}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n  Stopping streaming query...")
        query.stop()
        spark.stop()
        print("  PySpark consumer stopped.")


if __name__ == "__main__":
    main()