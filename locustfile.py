# =============================================================================
# locustfile.py — API Load Testing for EV Battery SOH Predictor
# Simulates 10k to 100k concurrent users hitting /predict and /analyze
#
# HOW TO RUN:
# -----------
# Step 1: Install locust
#         pip install locust
#
# Step 2: Make sure your FastAPI server is running
#         uvicorn main:app --host 0.0.0.0 --port 8000
#
# Step 3: Run locust with web UI (recommended for first time)
#         locust -f locustfile.py --host=http://localhost:8000
#         Then open http://localhost:8089 in browser
#         Set users=1000, spawn rate=50, and click Start
#
# Step 4: Run locust headless (for CI/CD or scripted tests)
#         Simulate 1000 users:
#         locust -f locustfile.py --host=http://localhost:8000
#               --headless -u 1000 -r 50 --run-time 2m
#
#         Simulate 10,000 users:
#         locust -f locustfile.py --host=http://localhost:8000
#               --headless -u 10000 -r 200 --run-time 5m
#
#         Simulate 100,000 users (needs powerful machine or distributed):
#         locust -f locustfile.py --host=http://localhost:8000
#               --headless -u 100000 -r 1000 --run-time 10m
#
# Step 5: Distributed load testing (for 100k users)
#         # On master machine:
#         locust -f locustfile.py --master --host=http://localhost:8000
#         # On each worker machine:
#         locust -f locustfile.py --worker --master-host=<master_ip>
#
# PARAMETERS EXPLAINED:
# -u  = total number of users to simulate
# -r  = users spawned per second (ramp-up rate)
# --run-time = how long to run the test (2m, 5m, 10m etc)
# =============================================================================

import io
import csv
import json
import random
import string
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner

# ---------------------------------------------------------------------------
# Sample battery data that mimics a real CSV upload
# We pre-generate this once so every simulated user reuses it
# without re-creating it on every request (saves CPU during load test)
# ---------------------------------------------------------------------------

# 52 rows gives us exactly 2 prediction windows (window_size=50, so 52-50=2)
SAMPLE_ROWS = 52

def generate_battery_csv(battery_id: int = 54, start_capacity: float = 0.744) -> bytes:
    """
    Generate a realistic battery CSV payload in memory.
    Returns raw bytes so it can be sent as a multipart file upload.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row — must match what FastAPI expects
    writer.writerow([
        "battery_id", "test_id", "Capacity",
        "Re", "Rct", "ambient_temperature"
    ])

    for i in range(SAMPLE_ROWS):
        # Simulate gradual capacity fade + small random noise
        capacity = round(start_capacity - (i * 0.002) + random.uniform(-0.001, 0.001), 6)
        re       = round(0.0145 + (i * 0.0002) + random.uniform(-0.00005, 0.00005), 6)
        rct      = round(0.0532 + (i * 0.0003) + random.uniform(-0.0001, 0.0001), 6)
        temp     = random.choice([4.0, 24.0])

        writer.writerow([battery_id, i + 1, capacity, re, rct, temp])

    return output.getvalue().encode("utf-8")


# Pre-generate payloads for a pool of 20 different batteries
# Real users would have different batteries — this simulates that diversity
BATTERY_PAYLOAD_POOL = [
    generate_battery_csv(
        battery_id=random.randint(40, 60),
        start_capacity=round(random.uniform(0.65, 0.80), 3)
    )
    for _ in range(20)
]


# ---------------------------------------------------------------------------
# Locust User Class — simulates one concurrent user
# ---------------------------------------------------------------------------

class BatteryAPIUser(HttpUser):
    """
    Simulates a single user of the EV Battery SOH API.

    wait_time: between(1, 3) means each user waits 1-3 seconds between tasks.
    This is realistic — a real EV monitoring system would poll every few seconds,
    not thousands of times per second per vehicle.

    To stress test more aggressively, lower this to between(0.1, 0.5).
    """
    wait_time = between(1, 3)

    def on_start(self):
        """
        Called once when a simulated user starts.
        Each user picks a random CSV payload from the pre-built pool
        so requests are not all identical (more realistic).
        """
        self.csv_payload = random.choice(BATTERY_PAYLOAD_POOL)

    # -----------------------------------------------------------------------
    # Task 1: POST /predict
    # Weight=3 means this task runs 3x more often than /analyze
    # This is realistic — most users just want predictions, fewer want AI analysis
    # -----------------------------------------------------------------------
    @task(3)
    def predict_soh(self):
        """
        Simulates a user uploading battery data for SOH prediction.
        Sends a multipart file upload to /predict endpoint.
        """
        files = {
            "file": ("battery_data.csv", self.csv_payload, "text/csv")
        }

        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,   # allows us to manually mark pass/fail
            name="/predict"        # groups all /predict calls in Locust UI
        ) as response:

            if response.status_code == 200:
                data = response.json()

                # Validate response has expected fields
                if "predictions" not in data or "mae" not in data:
                    response.failure(
                        f"Missing fields in response: {list(data.keys())}"
                    )
                elif data.get("total_cycles", 0) == 0:
                    response.failure("Got zero predictions — model may have failed")
                else:
                    response.success()

            elif response.status_code == 422:
                # Validation error — our CSV format is wrong
                response.failure(f"Validation error: {response.text[:200]}")

            elif response.status_code == 500:
                response.failure(f"Server error on /predict: {response.text[:200]}")

            else:
                response.failure(f"Unexpected status: {response.status_code}")

    # -----------------------------------------------------------------------
    # Task 2: POST /analyze (includes LangGraph + Azure OpenAI agent)
    # Weight=1 means this runs less often — AI calls are expensive and slower
    # -----------------------------------------------------------------------
    @task(1)
    def analyze_with_agent(self):
        """
        Simulates a user requesting full AI analysis including LangGraph agent.
        This endpoint is slower due to Azure OpenAI calls.
        We set a longer timeout (60s) to account for LLM latency.
        """
        files = {
            "file": ("battery_data.csv", self.csv_payload, "text/csv")
        }

        with self.client.post(
            "/analyze",
            files=files,
            catch_response=True,
            timeout=60,            # LLM calls can take 5-15 seconds
            name="/analyze (AI Agent)"
        ) as response:

            if response.status_code == 200:
                data = response.json()

                # Check AI analysis was actually returned
                if "ai_analysis" not in data:
                    response.failure("Missing ai_analysis in response")
                elif "OpenAI API key not provided" in data.get("ai_analysis", ""):
                    # API key missing — still mark as success since API itself worked
                    response.success()
                else:
                    response.success()

            elif response.status_code == 500:
                response.failure(f"Server error on /analyze: {response.text[:200]}")

            else:
                response.failure(f"Status {response.status_code}: {response.text[:100]}")

    # -----------------------------------------------------------------------
    # Task 3: GET / (health check)
    # Weight=1 — some monitoring systems ping health endpoints frequently
    # -----------------------------------------------------------------------
    @task(1)
    def health_check(self):
        """
        Simulates a load balancer or monitoring system checking API health.
        This should always be near-instant and helps identify if the server
        is up but slow (health check passes but /predict is slow).
        """
        with self.client.get(
            "/",
            catch_response=True,
            name="/  (health check)"
        ) as response:

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "running":
                    response.success()
                else:
                    response.failure(f"Unexpected health response: {data}")
            else:
                response.failure(f"Health check failed: {response.status_code}")


# ---------------------------------------------------------------------------
# Heavy User class — simulates power users / automated EV fleet systems
# These users hit the API much more aggressively (no wait time)
# ---------------------------------------------------------------------------

class EVFleetSystemUser(HttpUser):
    """
    Simulates an automated EV fleet management system — no human wait time.
    These fire requests as fast as possible, like a real IoT data pipeline.

    Use weight parameter to control how many of these vs normal users:
    Run with: --users 1000 and this class will be 10% of load by default.
    """
    wait_time = between(0.1, 0.5)   # much faster than human users
    weight    = 1                    # 1 weight vs BatteryAPIUser default weight of 3

    def on_start(self):
        self.csv_payload = random.choice(BATTERY_PAYLOAD_POOL)

    @task
    def rapid_predict(self):
        """
        Fires /predict as fast as possible — simulates an automated EV telemetry system
        sending battery readings every few hundred milliseconds.
        """
        files = {"file": ("data.csv", self.csv_payload, "text/csv")}

        with self.client.post(
            "/predict",
            files=files,
            catch_response=True,
            name="/predict (fleet system)"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Fleet predict failed: {response.status_code}")


# ---------------------------------------------------------------------------
# Event hooks — print a summary when test finishes
# ---------------------------------------------------------------------------

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Called when the load test completes.
    Prints a final performance summary to the console.
    """
    stats = environment.stats.total

    print("\n" + "=" * 60)
    print("  LOAD TEST COMPLETE — PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"  Total requests      : {stats.num_requests:,}")
    print(f"  Total failures      : {stats.num_failures:,}")
    print(f"  Failure rate        : {(stats.num_failures / max(stats.num_requests, 1)) * 100:.2f}%")
    print(f"  Avg response time   : {stats.avg_response_time:.0f} ms")
    print(f"  95th percentile     : {stats.get_response_time_percentile(0.95):.0f} ms")
    print(f"  99th percentile     : {stats.get_response_time_percentile(0.99):.0f} ms")
    print(f"  Max response time   : {stats.max_response_time:.0f} ms")
    print(f"  Requests/second     : {stats.current_rps:.1f}")
    print("=" * 60)

    # Verdict
    failure_rate = (stats.num_failures / max(stats.num_requests, 1)) * 100
    p95          = stats.get_response_time_percentile(0.95)

    if failure_rate < 1 and p95 < 2000:
        print("  VERDICT: PASSED — API handles load within acceptable limits")
    elif failure_rate < 5 and p95 < 5000:
        print("  VERDICT: WARNING — Some degradation under load, consider optimization")
    else:
        print("  VERDICT: FAILED — API cannot handle this load level")
    print("=" * 60 + "\n")