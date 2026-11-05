"""
This module handles hardware monitoring resources for the efficient evaluation of model inference, focusing on GPU power usage.
It provides threaded monitoring capabilities to track real-time power consumption metrics via the NVML interface.
The collected data allows for precise energy accounting per inference task, supporting green AI optimization efforts.
These utilities enable the correlation of latency and throughput metrics with actual hardware energy costs.
"""

import time
import torch
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

def get_instant_power(handle):
    try:
        power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)
        return power_mW / 1000.0
    except:
        return 0.0

def power_monitor_thread(handle, stop_event, readings_list, interval=0.05):
    while not stop_event.is_set():
        w = get_instant_power(handle)
        readings_list.append(w)
        time.sleep(interval)

def get_cache_memory_size(cache_instance) -> dict:
    total_bytes = 0
    if hasattr(cache_instance, 'layers'):
        for layer in cache_instance.layers:
            if hasattr(layer, 'is_initialized') and layer.is_initialized:
                total_bytes += layer.keys.nbytes + layer.values.nbytes
    return {"bytes": total_bytes, "MB": total_bytes / (1024 ** 2), "GB": total_bytes / (1024 ** 3)}

def init_nvml():
    handle = None
    if HAS_NVML:
        try:
            pynvml.nvmlInit()
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception:
            pass
    return HAS_NVML, handle