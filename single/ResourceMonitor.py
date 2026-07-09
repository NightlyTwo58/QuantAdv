import math
import os
import time
import threading

import psutil
import torch


class ResourceMonitor:
    """Record resource use around an existing run without executing extra model work."""

    def __init__(self, model, name, sample_interval_seconds=0.1):
        self.model = model
        self.name = name
        self.sample_interval_seconds = sample_interval_seconds
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.sampler = None
        self.peak_rss_bytes = 0
        self.metrics = None

    @staticmethod
    def _process_tree_usage(process):
        """Return RSS and CPU seconds for this process and its current children."""
        rss_bytes = 0
        user_seconds = 0.0
        system_seconds = 0.0
        try:
            processes = [process, *process.children(recursive=True)]
        except (psutil.Error, OSError):
            processes = [process]
        for proc in processes:
            try:
                rss_bytes += proc.memory_info().rss
                cpu = proc.cpu_times()
                user_seconds += cpu.user
                system_seconds += cpu.system
            except (psutil.Error, OSError):
                continue
        return rss_bytes, user_seconds, system_seconds

    def _sample_memory(self):
        while not self.stop_event.wait(self.sample_interval_seconds):
            rss_bytes, _, _ = self._process_tree_usage(self.process)
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)

    def __enter__(self):
        self.start_rss_bytes, self.start_user_seconds, self.start_system_seconds = (
            self._process_tree_usage(self.process)
        )
        self.peak_rss_bytes = self.start_rss_bytes
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_cuda_allocated = torch.cuda.memory_allocated()
            self.start_cuda_reserved = torch.cuda.memory_reserved()
        else:
            self.start_cuda_allocated = None
            self.start_cuda_reserved = None
        self.start_time = time.perf_counter()
        self.sampler = threading.Thread(target=self._sample_memory, daemon=True)
        self.sampler.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - self.start_time
        self.stop_event.set()
        self.sampler.join()
        end_rss_bytes, end_user_seconds, end_system_seconds = self._process_tree_usage(
            self.process
        )
        self.peak_rss_bytes = max(self.peak_rss_bytes, end_rss_bytes)

        params = list(self.model.parameters())
        parameter_count = sum(p.numel() for p in params)
        resident_bytes = sum(p.numel() * p.element_size() for p in params)
        quant_bits = [
            module.bits
            for module in self.model.modules()
            if getattr(module, "bits", None) is not None
            and hasattr(module, "quant_weight")
            and hasattr(module, "quant_act")
        ]
        nominal_bits = min(quant_bits) if quant_bits else 32
        packed_bytes = math.ceil(parameter_count * nominal_bits / 8)

        self.metrics = {
            "model": self.name,
            "status": "failed" if exc_type is not None else "completed",
            "run_seconds": elapsed_seconds,
            "cpu_user_seconds": max(0.0, end_user_seconds - self.start_user_seconds),
            "cpu_system_seconds": max(
                0.0, end_system_seconds - self.start_system_seconds
            ),
            "average_cpu_cores_used": (
                max(
                    0.0,
                    end_user_seconds
                    + end_system_seconds
                    - self.start_user_seconds
                    - self.start_system_seconds,
                )
                / elapsed_seconds
                if elapsed_seconds
                else None
            ),
            "rss_start_mib": self.start_rss_bytes / 2**20,
            "rss_end_mib": end_rss_bytes / 2**20,
            "rss_peak_mib": self.peak_rss_bytes / 2**20,
            "rss_peak_increase_mib": (self.peak_rss_bytes - self.start_rss_bytes)
            / 2**20,
            "parameter_count": parameter_count,
            "resident_model_mib": resident_bytes / 2**20,
            "nominal_packed_model_mib": packed_bytes / 2**20,
            "nominal_weight_bits": nominal_bits,
            "cuda_allocated_start_mib": (
                self.start_cuda_allocated / 2**20
                if self.start_cuda_allocated is not None
                else None
            ),
            "cuda_allocated_peak_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else None
            ),
            "cuda_reserved_start_mib": (
                self.start_cuda_reserved / 2**20
                if self.start_cuda_reserved is not None
                else None
            ),
            "cuda_reserved_peak_mib": (
                torch.cuda.max_memory_reserved() / 2**20
                if torch.cuda.is_available()
                else None
            ),
        }
        return False


ResourceMoniter = ResourceMonitor
