import time


class Metrics:
    def __init__(self):
        self.total_tokens_generated = 0
        self.start_time = time.time()
        self.current_batch_size = 0
        self.active_request_count = 0

    def log_batch(self, batch_size: int, active_count: int):
        self.current_batch_size = batch_size
        self.active_request_count = active_count

    def log_token(self):
        self.total_tokens_generated += 1

    def get_tokens_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_tokens_generated / elapsed if elapsed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "active_requests": self.active_request_count,
            "current_batch_size": self.current_batch_size,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_second": round(self.get_tokens_per_second(), 2),
        }
