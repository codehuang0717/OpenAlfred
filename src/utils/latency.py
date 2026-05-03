import time
from utils.logger import get_logger

logger = get_logger("latency")

class LatencyTracker:
    def __init__(self):
        self.timings = {}

    def start(self, name):
        self.timings[name] = {"start": time.perf_counter()}

    def end(self, name):
        if name in self.timings:
            self.timings[name]["end"] = time.perf_counter()
            self.timings[name]["duration"] = (
                self.timings[name]["end"] - self.timings[name]["start"]
            )

    def get(self, name):
        return self.timings.get(name, {}).get("duration", 0)

    def log_summary(self):
        logger.info("=== Latency Summary ===")
        for name, data in self.timings.items():
            logger.info(f"  {name}: {data.get('duration', 0) * 1000:.0f}ms")
        logger.info("=======================")

    def reset(self):
        self.timings = {}

# Singleton instance for the worker
latency_tracker = LatencyTracker()
