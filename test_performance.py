import os
import sys
import time
import statistics
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.backend.main import app

client = TestClient(app)

def run_latency_benchmark(num_requests=100, batch_size=10):
    print(f"\n=========================================================")
    print(f"   SMARTWEALTH AI - API Performance & Latency Benchmark  ")
    print(f"   Target Latency: < 120 ms per batch query              ")
    print(f"=========================================================\n")

    sample_txs = [
        {"description": f"WALMART SUPERCENTER #{i} BENTONVILLE", "amount": 45.0 + i}
        for i in range(batch_size)
    ]
    payload = {"transactions": sample_txs}

    latencies_ms = []

    # Warmup request
    client.post("/api/v1/predict/category", json=payload)

    for idx in range(num_requests):
        t0 = time.time()
        resp = client.post("/api/v1/predict/category", json=payload)
        t1 = time.time()
        
        assert resp.status_code == 200, f"Request {idx} failed with code {resp.status_code}"
        elapsed_ms = (t1 - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

    avg_lat = statistics.mean(latencies_ms)
    p95_lat = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
    p99_lat = sorted(latencies_ms)[int(0.99 * len(latencies_ms))]
    min_lat = min(latencies_ms)
    max_lat = max(latencies_ms)

    print(f"Executed {num_requests} batch queries (Batch size: {batch_size}):")
    print(f"  * Min Latency:       {min_lat:.2f} ms")
    print(f"  * Mean Latency:      {avg_lat:.2f} ms (Target: < 120.0 ms)")
    print(f"  * 95th Percentile:   {p95_lat:.2f} ms")
    print(f"  * 99th Percentile:   {p99_lat:.2f} ms")
    print(f"  * Max Latency:       {max_lat:.2f} ms")

    if avg_lat < 120.0:
        print(f"\n[OK] PASSED PERFORMANCE REQUIREMENT: Average batch latency ({avg_lat:.2f} ms) is well below 120 ms!")
    else:
        raise ValueError(f"[FAIL] FAILED PERFORMANCE REQUIREMENT: Average latency {avg_lat:.2f} ms exceeds 120 ms threshold!")

    return {
        "num_requests": num_requests,
        "batch_size": batch_size,
        "mean_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "target_met": avg_lat < 120.0
    }

if __name__ == "__main__":
    run_latency_benchmark(num_requests=100, batch_size=10)
