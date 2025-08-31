"""
HMM Prometheus Exporter - Export HMM metrics to Prometheus.
"""

import os
import time
import requests
from prometheus_client import start_http_server, Gauge, Counter

HMM_CORE_URL = os.getenv("HMM_CORE_URL", "http://localhost:8080")
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9091"))

hmm_state_prob = Gauge('hmm_state_probability', 'HMM state probability', ['state'])
hmm_detection_latency = Gauge('hmm_detection_latency_seconds', 'HMM detection latency')
hmm_cache_hits = Counter('hmm_cache_hits_total', 'HMM cache hits')
hmm_errors = Counter('hmm_errors_total', 'HMM detection errors')

def collect_metrics():
    """Collect metrics from HMM core service."""
    try:
        response = requests.get(f"{HMM_CORE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            metrics_text = response.text
            
            for line in metrics_text.split('\n'):
                if line.startswith('hmm_state_prob'):
                    parts = line.split()
                    if len(parts) >= 2:
                        state = parts[0].split('state="')[1].split('"')[0]
                        value = float(parts[1])
                        hmm_state_prob.labels(state=state).set(value)
                
                elif line.startswith('hmm_detection_latency_ms'):
                    parts = line.split()
                    if len(parts) >= 2:
                        latency_ms = float(parts[1])
                        hmm_detection_latency.set(latency_ms / 1000.0)
                
                elif line.startswith('hmm_cache_hit_rate'):
                    parts = line.split()
                    if len(parts) >= 2:
                        hit_rate = float(parts[1])
                        hmm_cache_hits._value._value = hit_rate * 100
        
    except Exception as e:
        print(f"Error collecting metrics: {e}")
        hmm_errors.inc()

def main():
    """Main exporter loop."""
    print(f"Starting HMM Prometheus exporter on port {PROMETHEUS_PORT}")
    start_http_server(PROMETHEUS_PORT)
    
    while True:
        collect_metrics()
        time.sleep(10)

if __name__ == "__main__":
    main()
