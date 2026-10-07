"""Process memory reading and a watchdog that stops the process above a cap (Windows reading; zeros elsewhere)."""

import ctypes
import os
import threading
import time
from ctypes import wintypes


def memory_mb() -> tuple:
    """(current, peak) working set in MB; Windows only, (0, 0) elsewhere."""
    if os.name != "nt":
        return 0, 0

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]

    kernel, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    c = Counters()
    c.cb = ctypes.sizeof(c)
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize // 2**20, c.PeakWorkingSetSize // 2**20


def guard(cap_gb: float) -> None:
    """Stop this process if its memory passes ``cap_gb``, so a mistake cannot take the machine down."""
    def watch() -> None:
        while True:
            if memory_mb()[0] > cap_gb * 1024:
                print(f"\nSTOPPED: memory passed {cap_gb} GB", flush=True)
                os._exit(3)
            time.sleep(0.3)

    threading.Thread(target=watch, daemon=True).start()
