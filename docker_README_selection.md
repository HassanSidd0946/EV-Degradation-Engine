# Docker Deployment Guide

> **Ek command se poora stack**: FastAPI + Redis + Kafka + ZooKeeper — sab Docker Compose ke zariye.

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Docker Desktop | 24+ | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Docker Compose | v2.x (included in Desktop) | — |

> **Note:** Java, Kafka binaries, aur Redis alag se install karne ki zaroorat nahi — sab containers mein aa jayega.

---

## Quick Start (3 steps)

### Step 1 — `.env` file banao

```bash
cp .env.example .env
# .env mein apni Azure OpenAI credentials fill karo
```

`.env` example:
```env
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_MODEL_NAME=gpt-4
AZURE_OPENAI_API_VERSION=2024-02-01
```

> **Important:** `.env` file kabhi commit mat karo. `.gitignore` mein already hai.

---

### Step 2 — Core stack start karo

```bash
docker-compose up --build
```

Yeh command in services ko start karta hai:
- `ev_api` — FastAPI ML inference server → `http://localhost:8000`
- `ev_redis` — Redis Streams buffer
- `ev_zookeeper` — Kafka coordination
- `ev_kafka` — Kafka broker (topic auto-create hoga)

---

### Step 3 — API test karo

```bash
# Health check
curl http://localhost:8000/

# SOH prediction (test CSV ke saath)
curl -X POST http://localhost:8000/predict \
  -F "file=@test_battery.csv"

# AI-powered analysis
curl -X POST http://localhost:8000/analyze \
  -F "file=@test_battery.csv"
```

API docs: `http://localhost:8000/docs`

---

## Optional Profiles

### Live Redis Dashboard (WebSocket)

```bash
docker-compose --profile streaming up
```

Starts the `stream-simulator` service jo CSV data ko Redis Streams mein push karta hai.
Dashboard: `http://localhost:3000` (pehle `python -m http.server 3000` run karo locally)

### Kafka Streaming Pipeline

```bash
docker-compose --profile kafka up
```

Starts:
- `ev_kafka_streamer` — CSV → Kafka topic `ev_battery_telemetry`
- `ev_kafka_consumer` — Kafka → per-battery buffer → TCN inference

---

## Architecture (Dockerized)

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

## Useful Commands

```bash
# Sab services band karo
docker-compose down

# Data bhi delete karo (volumes)
docker-compose down -v

# Ek service ki logs dekho
docker-compose logs -f api

# Running containers check karo
docker-compose ps

# API container ke andar jao
docker exec -it ev_api bash

# Sirf API rebuild karo
docker-compose up --build api
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Port 8000 already in use` | `lsof -i :8000` se process dhundo aur band karo |
| `Kafka health check failing` | 30-40 sec wait karo, Kafka slow start hota hai |
| `Model file not found` | `best_tcn_v2.keras` repo root mein hona chahiye |
| `Azure API timeout` | `.env` mein credentials check karo |
| `/analyze` slow hai | Expected — Azure OpenAI network call hai, ~5-15 sec normal hai |

---

## Production Notes

- `--workers 2` Dockerfile mein set hai — CPU cores ke hisaab se badhao
- CORS `allow_origins` `main.py` mein specific domain pe set karo production ke liye
- `best_tcn_v2.keras` ko Git LFS ya Azure Blob Storage mein move karo
- Secrets management ke liye Docker Secrets ya Azure Key Vault use karo