"""GPU board power sampling through NVML (Section 4.5).

A background thread polls board power every 20 ms, independently of the
forward passes, because the power sensor refreshes more slowly than one pass.
Energy = mean board power x elapsed time.
"""
import threading
import time

import numpy as np

try:
    import pynvml
except ImportError:  # energy measurement is optional
    pynvml = None


def gpu_handle(index=0):
    if pynvml is None:
        raise RuntimeError("pynvml is not installed: pip install nvidia-ml-py")
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    name = pynvml.nvmlDeviceGetName(handle)
    return handle, name.decode() if isinstance(name, bytes) else name


class PowerSampler:
    def __init__(self, handle, interval_s=0.02):
        self.handle, self.interval_s = handle, interval_s
        self.samples, self._stop, self._thread = [], threading.Event(), None

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def start(self):
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop sampling and return mean board power in watts."""
        self._stop.set()
        self._thread.join()
        return float(np.mean(self.samples)) if self.samples else 0.0
