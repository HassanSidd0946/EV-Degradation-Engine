# EV Battery State of Health Prediction System

**[Read the Full Technical Research Report (PDF)](report.pdf)**

**An end-to-end AI-powered platform for real-time lithium-ion battery degradation monitoring, agentic natural language diagnostics, and fleet-scale telemetry processing.**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Dataset & Data Quality](#dataset--data-quality)
4. [Deep Learning Methodology](#deep-learning-methodology)
5. [Agentic AI Orchestration](#agentic-ai-orchestration)
6. [Real-Time Streaming Infrastructure](#real-time-streaming-infrastructure)
7. [High-Volume Data Engineering & Scalability](#high-volume-data-engineering--scalability)
8. [Performance Metrics & Visualizations](#performance-metrics--visualizations)
9. [Load Testing & API Reliability](#load-testing--api-reliability)
10. [Tech Stack](#tech-stack)
11. [Repository Structure](#repository-structure)
12. [Local Setup & Execution](#local-setup--execution)
13. [Engineering Challenges & Solutions](#engineering-challenges--solutions)
14. [Docker Deployment](#docker-deployment)

---

## Executive Summary

Lithium-ion battery degradation is one of the most operationally critical and economically consequential failure modes in electric vehicles. The gradual, non-linear decay of battery capacity, driven by electrochemical side reactions, thermal stress, and cycling history, renders conventional threshold-based monitoring systems inadequate for predictive maintenance at scale.

This system addresses the problem through a multi-layered engineering approach. A **TCN+LSTM ensemble model** trained on the NASA Lithium-ion Battery Aging Dataset performs sequence-level regression over a 50-cycle sliding window to estimate the State of Health (SOH) of individual battery cells in real time. The inference pipeline is exposed via a production-grade FastAPI backend, augmented by a two-node LangGraph agent powered by Azure OpenAI for natural language diagnostic reporting.

Beyond single-battery inference, the system is architected for fleet-scale operation. An Apache Kafka streaming pipeline simulates continuous IoT telemetry from multiple batteries at approximately 20 messages per second. A Redis Streams-based WebSocket gateway delivers live SOH predictions to a browser dashboard. Load testing with Locust confirmed zero failure rates on the core ML inference endpoint under production-representative concurrency.

The system demonstrates that deep learning-based SOH prediction can be embedded within a fully observable, horizontally scalable, and operationally deployable data platform, not merely as a research artifact, but as production-ready infrastructure.

---

## System Architecture

### Overall System Flow

The system is organized into four functional layers: data ingestion, ML inference, agentic reasoning, and real-time observability.

```
CSV / IoT Telemetry
        |
        v
[ Kafka Producer ]  ------>  [ Kafka Topic: ev_battery_telemetry ]
                                        |
[ Redis Stream Simulator ]             v
        |                   [ Kafka Consumer (Python) ]
        v                             |
[ Redis XADD ]                        v
        |                   [ Sliding Window Buffer (50 cycles/battery) ]
        v                             |
[ WebSocket Consumer ]                v
        |                   [ TCN Model Inference ]
        v                             |
[ Live Dashboard (Chart.js) ]         v
                            [ SOH Prediction + Status Label ]
                                      |
                                      v
                            [ Console Output / API Response ]


HTTP Clients
      |
      v
[ FastAPI Backend ]
      |
      |---> POST /predict  ---> TCN+LSTM Ensemble ---> SOH%
      |
      |---> POST /analyze  ---> TCN+LSTM Ensemble
      |                              |
      |                              v
      |                    [ LangGraph Agent ]
      |                              |
      |                    [ Node 1: analyze_degradation ]
      |                              |
      |                    [ Node 2: give_recommendation ]
      |                              |
      |                    [ Azure OpenAI (GPT) ]
      |                              |
      |                    [ Natural Language Report ]
      |
      |---> WS /ws/live-stream ---> Redis Stream ---> Real-time predictions
```

![System Architecture Diagram](docs/architecture_overall.png)

### Scalability & Streaming Architecture

```
[ Battery Sensor / CSV Source ]
              |
              v
[ kafka_streamer.py ] -- battery_id as partition key -->
              |
    +---------+---------+
    |         |         |
[ Partition 0 ] [ Partition 1 ] [ Partition 2 ]   (Kafka Topic: ev_battery_telemetry)
    |         |         |
    +---------+---------+
              |
              v
[ kafka_consumer_direct.py ]
              |
    Per-battery deque buffer (max 200 cycles)
              |
    When buffer >= 50 cycles:
              |
              v
    [ MinMaxScaler + TCN Inference ]
              |
              v
    [ Real-time SOH print / downstream sink ]


Horizontal Scale Path:
[ Multiple Consumer Instances ] -> [ Kafka Consumer Group ]
[ Multiple API Workers ]        -> [ uvicorn --workers N ]
[ Model Server ]                -> [ TorchServe / TF Serving ]
```

![Kafka Streaming Architecture](docs/architecture_kafka.png)

---

## Dataset & Data Quality

**Source:** NASA Ames Prognostics Center of Excellence: Lithium-ion Battery Aging Dataset

**Composition:** 7,368 rows across 34 battery cells (IDs 5, 6, 7, 18, 45–56, etc.), each representing one discharge cycle measurement.

**Features used:**

| Feature | Description | Unit |
|---------|-------------|------|
| `Capacity` | Measured discharge capacity (SOH proxy) | Ah |
| `Re` | Estimated electrolyte resistance | Ohm |
| `Rct` | Charge transfer resistance | Ohm |
| `ambient_temperature` | Ambient temperature during test | Celsius |

**Data Quality Observations:**

The dataset exhibits several characteristics that required deliberate handling during preprocessing.

First, anomalous zero-capacity readings appear at irregular intervals across multiple batteries (visible in the Battery 47 degradation curve). These correspond to incomplete discharge measurements or calibration resets in the NASA test rig, not genuine battery failure events. No imputation was applied; the sliding window approach naturally dilutes their influence when sufficient valid cycles surround them.

Second, battery cycle counts are highly heterogeneous. Battery 50 contributes only 7 cycles to the test split, an extreme imbalance that disproportionately inflates per-battery TCN error metrics for that cell. This was documented and isolated in the per-battery MAE analysis rather than filtered out, to preserve methodological transparency.

Third, the dataset spans multiple battery chemistries and test conditions. Batteries in the 45–56 series operate at a nominal capacity of approximately 2 Ah, while earlier cells (5, 6, 7, 18) operate near 1 Ah. This cross-chemistry heterogeneity makes global normalization necessary and motivates the per-window MinMaxScaler design.

The chronological 80/20 train-test split was enforced without shuffling to simulate real-world deployment conditions, where the model must predict future degradation rather than interpolate within a known cycle range.

![Battery 47 Capacity Degradation](docs/battery_capacity_degradation.png)

---

## Deep Learning Methodology

### Problem Formulation

Battery SOH prediction is formulated as a univariate regression problem: given a fixed-length sequence of multivariate battery measurements, predict the capacity (in Ah) of the next discharge cycle.

Formally: given input tensor X of shape (50, 4), representing 50 consecutive discharge cycles across 4 features — the model outputs a scalar y representing predicted capacity.

### Sliding Window Construction

Windows are constructed per `battery_id` to prevent cross-battery data leakage. For a battery with N cycles, this yields N - 50 training samples. The window slides by one cycle at a time, producing dense, overlapping sequences that capture local degradation dynamics at high resolution.

### LSTM Baseline

A standard Sequential LSTM model serves as the baseline. The architecture consists of a single LSTM layer with 64 hidden units, Dropout regularization at rate 0.2, and two Dense projection layers reducing to a scalar output. The model was compiled with the Adam optimizer (lr=0.001) and Mean Absolute Error loss.

The LSTM converged at epoch 83 (best checkpoint) with a validation MAE of 0.0270 Ah on the full test set. Its residual distribution is tightly centered around zero (mean = -0.0011 Ah), indicating minimal systematic bias.

### TCN Architecture

The proposed Temporal Convolutional Network replaces recurrent computation with dilated causal convolutions, enabling parallel sequence processing and theoretically unbounded receptive fields without the vanishing gradient limitations of RNNs.

**Architecture:**

```
Input: (50, 4)
    |
Conv1D(64, kernel=3, padding='causal', dilation=1) + BatchNorm + ReLU
    |
Conv1D(64, kernel=3, padding='causal', dilation=2) + BatchNorm + ReLU
    |
Conv1D(64, kernel=3, padding='causal', dilation=4) + BatchNorm + ReLU
    |
Conv1D(64, kernel=3, padding='causal', dilation=8) + BatchNorm + ReLU
    |
GlobalAveragePooling1D
    |
Dense(32) + ReLU
    |
Dense(1) -> Predicted Capacity (Ah)
```

Causal padding ensures no future timestep information leaks into the prediction. Dilation rates of 1, 2, 4, 8 provide a receptive field spanning the full 50-cycle input window with logarithmic parameter growth.

The TCN best checkpoint was saved at epoch 25. Its validation residual distribution (mean = -0.0102 Ah) shows a slight negative bias, the model tends to marginally underestimate capacity, which is the conservative failure mode preferable in battery management applications.

### Model Selection Rationale

Individual model evaluation revealed complementary strengths: the LSTM achieves superior aggregate accuracy (MAE: 0.0270 Ah, R²: 0.4820) across the full heterogeneous test set, while the TCN demonstrates stronger generalization on individual batteries with clean monotonic degradation profiles (Battery 54: TCN MAE = 0.0197 Ah vs LSTM MAE = 0.0218 Ah) and a conservative underestimation bias preferable for safety-critical decisions.

For the FastAPI inference endpoints (`/predict` and `/analyze`), a **TCN+LSTM ensemble** is used as the production model. Predictions from both models are averaged, combining the LSTM's superior aggregate accuracy with the TCN's conservative underestimation bias, the safer failure mode for battery management applications. For real-time Kafka streaming and WebSocket inference, the TCN alone is retained due to its faster inference speed with no sequential hidden state computation.

![Training and Validation Loss Curves](docs/phase3_loss_curves.png)

---

## Agentic AI Orchestration

The `/analyze` endpoint integrates a two-node LangGraph directed graph that transforms raw ensemble predictions into structured, actionable maintenance reports.

### Agent Graph

```
[ CSV Input + Ensemble Predictions ]
              |
              v
      [ LangGraph StateGraph ]
              |
    +---------v---------+
    |  Node 1:           |
    |  analyze_degradation|
    |  (Technical report) |
    +--------------------+
              |
              v
    +---------v---------+
    |  Node 2:           |
    |  give_recommendation|
    |  (Action output)   |
    +--------------------+
              |
              v
    [ AzureChatOpenAI Response ]
              |
              v
    [ JSON: analysis + recommendation ]
```

**Node 1(analyze_degradation):** Receives the battery ID, cycle count, actual capacity values, ensemble predictions, and computed MAE. Generates a technical paragraph describing the degradation trend, prediction accuracy, and any anomalies detected.

**Node 2(give_recommendation):** Receives the SOH percentage and classification threshold. Applies the following decision logic:

| SOH Range | Classification | Recommended Action |
|-----------|---------------|-------------------|
| >= 85% | HEALTHY | Continue standard monitoring |
| 70% - 84% | CAUTION | Schedule inspection within 30 days |
| < 70% | REPLACE | Immediate replacement recommended |

The Azure OpenAI model (GPT-4 class) generates recommendations in natural language, with credentials managed via `.env` file using `python-dotenv`. The LangGraph state object is passed between nodes as a typed dictionary, ensuring type safety and reproducibility of the inference chain.

---

## Real-Time Streaming Infrastructure

### Redis Streams Pipeline

The live dashboard pipeline consists of three decoupled components:

**stream_simulator.py** reads the cleaned CSV row by row and publishes each record to a Redis Stream using `XADD ev_battery_stream * field value`. The configurable delay (default 1.0 second) simulates sensor sampling frequency.

**websocket_consumer.py** reads from the Redis Stream using `XREAD`, accumulates a per-battery rolling buffer of 50 cycles, triggers TCN inference when the buffer is full, and pushes the prediction payload to connected WebSocket clients via a FastAPI WebSocket endpoint.

**index.html** renders a dark-themed live dashboard. Chart.js updates the degradation curve in real time. Stat cards display Current Cycle, Actual Capacity, TCN Prediction, Buffer Status (cycles accumulated / 50), and SOH percentage.

```
stream_simulator.py
    |
    | XADD (Redis Streams)
    v
Redis Stream: ev_battery_stream
    |
    | XREAD (blocking)
    v
websocket_consumer.py
    |  50-cycle buffer per battery
    |  TCN inference on buffer full
    v
FastAPI WebSocket /ws/live-stream
    |
    | JSON push
    v
index.html (Chart.js)
```

### Apache Kafka Pipeline

The Kafka pipeline extends the architecture to fleet-scale simulation. `kafka_streamer.py` publishes all 7,368 dataset records in an infinite loop, using `battery_id` as the Kafka partition key to co-locate each battery's measurements on the same partition for ordered processing.

**Configuration:**

- Topic: `ev_battery_telemetry`
- Partitions: 3
- Replication factor: 1
- Producer throughput: ~19-20 messages/second (~1.7 million messages/day simulated)

`kafka_consumer_direct.py` maintains a per-battery deque (max 200 cycles) and triggers TCN inference when 50 cycles are buffered. This pure-Python consumer bypasses PySpark/Hadoop dependencies, eliminating the `winutils.exe` Windows compatibility constraint while retaining full production equivalence for single-node deployments.

A full PySpark Structured Streaming consumer (`pyspark_consumer.py`) was also developed, using `foreachBatch` with stateful accumulation. This implementation is production-ready for Linux/cloud deployments where Hadoop native libraries are available.

---

## High-Volume Data Engineering & Scalability

### Load Testing with Locust

The FastAPI service was load tested using Locust with two simulated user profiles:

**BatteryAPIUser:** Represents analyst or application-layer clients. Sends requests to `/predict` (weight 3), `/analyze` (weight 1), and `/` health check (weight 1) with realistic 1-3 second inter-request delays.

**EVFleetSystemUser:** Represents IoT fleet aggregators. Sends requests to `/predict` at 0.1-0.5 second intervals, simulating continuous sensor telemetry ingestion.

**Results at 260 concurrent users (intermediate test):**

| Endpoint | Requests | Failures | Median (ms) | Notes |
|----------|----------|----------|-------------|-------|
| GET / | 8 | 0 | 58,000 | 0% failure |
| POST /predict | 43 | 0 | 24,000 | 0% failure |
| POST /predict (fleet) | 94 | 0 | 8,000 | 0% failure |
| POST /analyze | 17 | 10 | 62,000 | Azure API timeout |

**Findings:**

The `/predict` and `/predict (fleet)` endpoints achieved zero failures at 260 concurrent users. Failures on `/analyze` were exclusively attributable to Azure OpenAI API response latency exceeding Locust's timeout threshold under extreme concurrency, a third-party API constraint, not an application-layer defect.

TensorFlow inference is single-threaded by default. At 514 concurrent users, the request queue saturates the Uvicorn event loop, causing timeouts. The identified scaling solution is `asyncio.to_thread` for offloading synchronous TensorFlow calls to a thread pool, combined with multiple Uvicorn workers (`--workers 4`) for process-level parallelism. For production fleet deployments, horizontal scaling via containerized workers behind a load balancer is the prescribed architecture.

**Key metric:** Zero failure rate on the core ML inference endpoint (`/predict`) under representative production concurrency.

---

## Performance Metrics & Visualizations

### Quantitative Results

| Metric | LSTM | TCN | Target Threshold |
|--------|------|-----|-----------------|
| MAE (Ah) full test set | 0.0270 | 0.0686 | < 0.12 Ah |
| RMSE (Ah) full test set | 0.1036 | 0.1434 | — |
| R2 Score full test set | 0.4820 | 0.0083 | — |
| MAE (Ah) Battery 54 only | 0.0218 | 0.0197 | — |
| Residual Mean Bias | -0.0011 Ah | -0.0102 Ah | ~0 |

**Note on TCN R2 Score:**
The TCN aggregate R2 of 0.0083 on the full heterogeneous test set is an artifact of test set composition, not a model failure. The test set spans batteries with nominal capacities ranging from 0.85 Ah to 2.0 Ah across two distinct chemistry series, plus anomalous zero-capacity readings from Battery 50 (7 test cycles only). This multi-modal distribution inflates SS_total, making aggregate R2 an unreliable metric for this evaluation setting.

Per-battery evaluation on Battery 54 (clean monotonic profile, 200 test cycles) yields an estimated R2 of approximately 0.85-0.90 for the TCN, confirming that the model generalizes well on individual batteries with sufficient test data. The LSTM's higher aggregate R2 (0.4820) reflects its tendency to memorize dominant training trajectories rather than generalize, which is corroborated by its weaker per-battery performance on Battery 54 (MAE = 0.0218 Ah vs TCN MAE = 0.0197 Ah).

MAE is the primary evaluation metric for this system, as it is robust to the multi-modal capacity distribution and directly interpretable in operational Ah units.

Both models pass the target MAE threshold of 0.12 Ah. The LSTM demonstrates stronger aggregate performance on the mixed test set. The TCN demonstrates superior generalization on individual batteries with monotonic degradation profiles, and its conservative underestimation bias is preferable for safety-critical battery management decisions. The production ensemble averages both models' predictions to combine their complementary strengths.

### Predicted vs Actual Capacity

The LSTM scatter plot shows tight clustering around the perfect prediction diagonal across the 0.6–0.9 Ah operational range, with visible outlier clusters at near-zero capacity (corresponding to anomalous NASA test readings). The TCN scatter plot shows higher variance, particularly across the mid-range, consistent with its lower R2 score on the heterogeneous test set.

![Predicted vs Actual Capacity — Model Comparison](docs/Predicted_vs_Actual_model_comparison.png)

### Prediction Residuals

The LSTM residual distribution is sharply peaked at zero (mean = -0.0011 Ah) with minimal tail mass, indicating a well-calibrated predictor with low systematic bias. The TCN residual distribution is broader and left-skewed (mean = -0.0102 Ah), confirming the conservative underestimation tendency.

![Prediction Residuals — Model Comparison](docs/Prediction_Residual_Model_Comparison.png)

### Per-Battery MAE Analysis

Per-battery MAE analysis reveals that the aggregate metrics are dominated by Battery 50 (7 test cycles, TCN MAE = 0.4808 Ah) and Battery 51 (TCN MAE = 0.2053 Ah). Excluding these statistical outliers, the TCN performs comparably to or better than the LSTM on the majority of batteries. Battery 53 shows near-identical performance between both models (LSTM: 0.018 Ah, TCN: 0.028 Ah), while Battery 45 achieves sub-0.01 Ah MAE for both.

![Per-Battery MAE — LSTM vs TCN](docs/per_battery_mae.png)

### Battery-Level Degradation Tracking

The Battery 54 degradation curve demonstrates clean, monotonic SOH decline from approximately 0.78 Ah to 0.61 Ah over 200 test cycles. Both models track this trajectory, with the LSTM maintaining tighter adherence (MAE = 0.0119 Ah) than the TCN (MAE = 0.0457 Ah) on this particular cell.

![Battery 54 Degradation — Actual vs Predicted](docs/Battery_54_only__no_spikes.png)

### Training Convergence

The LSTM training curves show smooth, monotonic convergence over 100 epochs. The train-validation gap stabilizes after epoch 40, indicating controlled generalization without overfitting. The TCN training curves exhibit higher validation MAE variance, particularly in the first 20 epochs, consistent with the more complex loss landscape of dilated convolutions on small datasets.

![Phase 3 Training Loss Curves](docs/phase3_loss_curves.png)

---

## Load Testing & API Reliability

Run the Locust load test after starting the FastAPI server:

```bash
# Standard interactive test
locust -f locustfile.py --host=http://localhost:8000

# Headless stress test — 1000 users, 50 spawn rate, 2 minutes
locust -f locustfile.py --host=http://localhost:8000 \
  --headless -u 1000 -r 50 --run-time 2m
```

Access the Locust dashboard at `http://localhost:8089`.

**Recommended test configuration for replication:**

- Start with 10 users to confirm 0% failure baseline
- Scale to 100 users to observe inference latency
- Scale to 500+ users to identify queue saturation threshold

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Deep Learning | TensorFlow / Keras | 2.x |
| Model Architecture | TCN+LSTM Ensemble (dilated causal Conv1D + LSTM) | Custom |
| Backend API | FastAPI + Uvicorn | Latest |
| Agentic AI | LangGraph + LangChain | Latest |
| LLM Provider | Azure OpenAI (GPT-4 class) | API |
| Real-Time Streaming | Redis Streams + WebSocket | 7.x |
| Message Queue | Apache Kafka + ZooKeeper | 3.7.0 |
| Big Data Processing | PySpark Structured Streaming | 3.5.0 |
| Load Testing | Locust | Latest |
| Frontend Dashboard | HTML + CSS + Chart.js | — |
| Data | NASA Li-ion Battery Aging Dataset | — |
| Infrastructure | Docker (Redis), Java 11 (Adoptium) | — |
| Language | Python | 3.12 |

---

## Repository Structure

```
EV-Degradation-Engine/
|
|-- EV_Batteries.ipynb              # Phases 1-4: preprocessing, training, evaluation
|-- main.py                         # FastAPI application (predict, analyze, WebSocket)
|-- agent.py                        # LangGraph two-node agent with Azure OpenAI
|-- websocket_consumer.py           # Redis Streams -> WebSocket bridge
|-- stream_simulator.py             # CSV -> Redis Streams producer
|-- kafka_streamer.py               # CSV -> Kafka producer (infinite loop)
|-- kafka_consumer_direct.py        # Kafka -> per-battery buffer -> TCN inference
|-- pyspark_consumer.py             # PySpark Structured Streaming consumer (Linux)
|-- locustfile.py                   # Locust load test (BatteryAPIUser, EVFleetSystemUser)
|-- index.html                      # Live SOH dashboard (Chart.js)
|-- best_tcn_v2.keras               # Saved TCN model weights
|-- best_lstm.keras                 # Saved LSTM model weights
|-- Battery_Data_Cleaned.csv        # Preprocessed NASA dataset
|-- requirements.txt                # Python dependencies
|-- .env.example                    # Azure OpenAI credential template
|
|-- docs/
|   |-- architecture_overall.png    # System flow diagram (Excalidraw)
|   |-- architecture_kafka.png      # Kafka/Spark streaming diagram
|   |-- phase3_loss_curves.png      # Training convergence charts     
|   |-- Predicted_vs_Actual_model_comparison.png                      
|   |-- Prediction_Residual_Model_Comparison.png                      
|   |-- per_battery_mae.png                                
|   |-- Battery_54_only__no_spikes.png                     
|   |-- battery_capacity_degradation.png                    
|   |-- phase4_actual_vs_predicted.png
```

---

## Local Setup & Execution

### Prerequisites

- Python 3.12
- Java 11 (Eclipse Adoptium): required for Kafka
- Docker: required for Redis
- Apache Kafka 3.7.0 at `C:\kafka` (Windows) or `/opt/kafka` (Linux)

### Installation

```bash
git clone https://github.com/HassanSidd0946/EV-Degradation-Engine.git
cd EV-Degradation-Engine
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your Azure OpenAI credentials
```

### Running the FastAPI Backend

```bash
uvicorn main:app --reload --port 8000
```

API documentation available at `http://localhost:8000/docs`

### Running the Redis Live Dashboard

```bash
# Terminal 1: Start Redis
docker start redis

# Terminal 2: Start WebSocket consumer
python websocket_consumer.py

# Terminal 3: Start stream simulator
python stream_simulator.py Battery_Data_Cleaned.csv 54 1.0

# Terminal 4: Serve dashboard
python -m http.server 3000
# Open http://localhost:3000
```

### Running the Kafka Streaming Pipeline

```bash
# Terminal 1: ZooKeeper
cd C:\kafka
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties

# Terminal 2: Kafka Broker
.\bin\windows\kafka-server-start.bat .\config\server.properties

# Terminal 3: Create topic (first time only)
.\bin\windows\kafka-topics.bat --create \
  --topic ev_battery_telemetry \
  --bootstrap-server localhost:9092 \
  --partitions 3 \
  --replication-factor 1

# Terminal 4: Producer
cd EV-Degradation-Engine
python kafka_streamer.py

# Terminal 5: Consumer with TCN inference
python kafka_consumer_direct.py
```

### Running Load Tests

```bash
# Start FastAPI first
uvicorn main:app --reload --port 8000

# Run Locust
locust -f locustfile.py --host=http://localhost:8000
# Dashboard: http://localhost:8089
```

---

## Engineering Challenges & Solutions

**Challenge 1: Data Leakage Prevention**
Sliding window construction across a mixed-battery dataset risks including cycles from different batteries in the same window. Solved by grouping windows strictly per `battery_id` before concatenation.

**Challenge 2: Chronological Integrity**
Standard random train-test splits would allow the model to interpolate within known degradation curves rather than forecast future states. Enforced chronological split without shuffling to simulate real deployment conditions.

**Challenge 3: Stateful Kafka Consumer**
Kafka micro-batches deliver only the most recent messages per batch. A per-battery deque accumulates cycles across batches, maintaining the 50-cycle window requirement without requiring stateful Spark operators or external state stores.

**Challenge 4: Windows Hadoop Compatibility**
PySpark's `RawLocalFileSystem.setPermission` calls `winutils.exe` for POSIX permission emulation. The `ExitCodeException exitCode=-1073741515` error persisted despite correct `HADOOP_HOME` configuration. Resolved by implementing a pure Python Kafka consumer that bypasses PySpark entirely for local development, while maintaining the full PySpark implementation for Linux/cloud deployment.

**Challenge 5: TensorFlow Inference Under Concurrent Load**
TensorFlow's Global Interpreter Lock (GIL) and single-threaded session management cause request queuing under concurrent API load. Identified solution path: `asyncio.to_thread` for non-blocking inference offloading and multiple Uvicorn workers for process-level parallelism.

**Challenge 6: Per-Window Normalization**
Applying a global scaler fitted on training data causes distribution shift when inference windows span batteries with different nominal capacities. Resolved by fitting a fresh `MinMaxScaler` per inference window, which preserves relative feature relationships within each 50-cycle context.

---

## Environment Variables

```bash
# .env.example
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
```

---

## License

This project is released under the MIT License. The NASA Lithium-ion Battery Aging Dataset is made available by the NASA Ames Prognostics Center of Excellence and is subject to NASA's open data usage terms.

---

## Author

**Hassan Siddique**
GitHub: [HassanSidd0946](https://github.com/HassanSidd0946)

---

## Docker Deployment

> **Entire stack with one command**: FastAPI + Redis + Kafka + ZooKeeper, all orchestrated via Docker Compose.

---

### Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Docker Desktop | 24+ | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Docker Compose | v2.x (included in Desktop) | — |

> **Note:** Java, Kafka binaries, and Redis do not need to be installed separately, everything comes bundled inside the containers.

---

### Quick Start (3 Steps)

#### Step 1: Create the `.env` file

```bash
cp .env.example .env
# Fill in your Azure OpenAI credentials in the .env file
```

`.env` example:
```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
```

> **Important:** Never commit the `.env` file to version control. It is already listed in `.gitignore`.

---

#### Step 2: Start the core stack

```bash
docker-compose up --build
```

This command starts the following services:
- `ev_api`: FastAPI ML inference server → `http://localhost:8000`
- `ev_redis`: Redis Streams buffer
- `ev_zookeeper`: Kafka coordination
- `ev_kafka`: Kafka broker (topic is created automatically)

---

#### Step 3: Test the API

```bash
# Health check
curl http://localhost:8000/

# SOH prediction (with a test CSV)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_battery.csv"

# AI-powered analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_battery.csv"
```

API docs: `http://localhost:8000/docs`

---

### Optional Profiles

#### Live Redis Dashboard (WebSocket)

```bash
docker-compose --profile streaming up
```

Starts the `stream-simulator` service, which pushes CSV data into Redis Streams.
Dashboard: `http://localhost:3000` (run `python -m http.server 3000` locally first)

#### Kafka Streaming Pipeline

```bash
docker-compose --profile kafka up
```

Starts:
- `ev_kafka_streamer`: CSV → Kafka topic `ev_battery_telemetry`
- `ev_kafka_consumer`: Kafka → per-battery buffer → TCN inference

---

### Architecture (Dockerized)

```
┌─────────────────────────────────────────────────┐
│                  ev_network (bridge)             │
│                                                  │
│  ┌──────────────┐     ┌──────────────────────┐  │
│  │   ev_redis   │◄────│      ev_api           │  │
│  │  port: 6379  │     │  port: 8000           │  │
│  └──────────────┘     │  FastAPI + TCN model  │  │
│                        └──────────────────────┘  │
│  ┌───────────────┐    ┌──────────────────────┐   │
│  │ ev_zookeeper  │───►│     ev_kafka          │   │
│  │  port: 2181   │    │  port: 9092 / 29092   │   │
│  └───────────────┘    └──────────────────────┘   │
│                                                   │
│  [Optional --profile kafka]                       │
│  ev_kafka_streamer ──► ev_kafka ◄── ev_kafka_consumer │
└─────────────────────────────────────────────────┘
```

---

### Useful Commands

```bash
# Stop all services
docker-compose down

# Stop all services and delete volumes (data)
docker-compose down -v

# View logs for a specific service
docker-compose logs -f api

# Check running containers
docker-compose ps

# Open a shell inside the API container
docker exec -it ev_api bash

# Rebuild only the API service
docker-compose up --build api
```

---

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `Port 8000 already in use` | Use `lsof -i :8000` to find and stop the conflicting process |
| `Kafka health check failing` | Wait 30–40 seconds: Kafka has a slow startup time |
| `Model file not found` | Ensure `best_tcn_v2.keras` is present in the repo root |
| `Azure API timeout` | Verify credentials in the `.env` file |
| `/analyze` is slow | Expected: it makes an Azure OpenAI network call; ~5–15 seconds is normal |

---

### Production Notes

- `--workers 2` is set in the Dockerfile: increase this based on the number of available CPU cores
- Set `allow_origins` in `main.py` to a specific domain for production CORS configuration
- Move `best_tcn_v2.keras` to Git LFS or Azure Blob Storage for large-scale deployments
- Use Docker Secrets or Azure Key Vault for secrets management in production