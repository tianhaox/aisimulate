# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only (not for merge): what the host looks like around one module-series task.

Host-bound points of some tasks are 1.6x slow as a whole on B300. One CPUDIAG line per task records where the
process ran, what else was busy next to it, and a fixed pure-Python benchmark, to tell a slow core (frequency,
SMT sibling) from contention inside the job.
"""

import json
import os
import statistics
import time


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _cpu_times():
    """cpu -> (busy jiffies, total jiffies) from /proc/stat."""
    out = {}
    for line in (_read("/proc/stat") or "").splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:9]]
            out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out


def _threads():
    """tid -> (comm, utime + stime jiffies, last cpu) of this process."""
    out = {}
    try:
        tids = os.listdir("/proc/self/task")
    except OSError:
        return out
    for tid in tids:
        s = _read(f"/proc/self/task/{tid}/stat")
        if not s:
            continue
        comm = s[s.index("(") + 1 : s.rindex(")")]
        f = s[s.rindex(")") + 2 :].split()
        out[int(tid)] = (comm, int(f[11]) + int(f[12]), int(f[36]))
    return out


def _status(*keys):
    out = {}
    for line in (_read("/proc/self/status") or "").splitlines():
        k, _, v = line.partition(":")
        if k in keys:
            out[k] = v.strip()
    return out


def _siblings(cpu):
    s = _read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") or ""
    out = set()
    for part in s.replace("-", ",").split(","):
        if part.strip().isdigit():
            out.add(int(part))
    return sorted(out - {cpu})


def bench_ms(reps=5):
    """A fixed single-thread pure-Python loop: (min, median) in ms."""
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        x = 0
        for i in range(1000000):
            x += i * i
        times.append((time.perf_counter() - t) * 1e3)
    return round(min(times), 3), round(statistics.median(times), 3)


def snapshot():
    cpu = os.sched_getcpu() if hasattr(os, "sched_getcpu") else -1
    return {
        "t": time.time(),
        "cpu": cpu,
        "sib": _siblings(cpu),
        "freq_khz": _read(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"),
        "bench": bench_ms(),
        "cpu_times": _cpu_times(),
        "threads": _threads(),
        "status": _status("Cpus_allowed_list", "voluntary_ctxt_switches", "nonvoluntary_ctxt_switches", "Threads"),
        "load": _read("/proc/loadavg"),
    }


def report(label, before, after):
    """Print one CPUDIAG json line for the task that ran between the two snapshots."""

    def share(cpu):
        b, a = before["cpu_times"].get(cpu), after["cpu_times"].get(cpu)
        if not b or not a or a[1] == b[1]:
            return None
        return round((a[0] - b[0]) / (a[1] - b[1]), 3)

    allowed = before["status"].get("Cpus_allowed_list")
    try:
        aff = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = []
    shares = {c: share(c) for c in after["cpu_times"]}
    busy_all = sum(1 for v in shares.values() if v is not None and v > 0.5)
    busy_aff = sum(1 for c in aff if (shares.get(c) or 0) > 0.5)
    hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    wall = after["t"] - before["t"]
    threads = []
    for tid, (comm, ticks, last) in after["threads"].items():
        d = ticks - before["threads"].get(tid, (comm, 0, last))[1]
        if d > 0:
            threads.append((round(d / hz / max(wall, 1e-9), 3), comm, tid == os.getpid(), last))
    threads.sort(reverse=True)
    ctx = {
        k: int(after["status"].get(k, 0)) - int(before["status"].get(k, 0))
        for k in ("voluntary_ctxt_switches", "nonvoluntary_ctxt_switches")
    }
    line = {
        "label": label,
        "pid": os.getpid(),
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "wall_s": round(wall, 1),
        "allowed": allowed,
        "n_aff": len(aff),
        "n_cpus": len(after["cpu_times"]),
        "cpu_before": before["cpu"],
        "cpu_after": after["cpu"],
        "sib_before": before["sib"],
        "sib_after": after["sib"],
        "own_share": share(after["cpu"]),
        "sib_share": [share(c) for c in after["sib"]],
        "busy_cpus_node": busy_all,
        "busy_cpus_allowed": busy_aff,
        "freq_khz": [before["freq_khz"], after["freq_khz"]],
        "bench_ms_before": before["bench"],
        "bench_ms_after": after["bench"],
        "threads_total": after["status"].get("Threads"),
        "top_threads": threads[:6],  # (cpu share of wall, name, is main thread, last cpu)
        "ctx_switches": ctx,
        "load": [before["load"], after["load"]],
        "cpu_max": _read("/sys/fs/cgroup/cpu.max"),
        "cpuset_effective": _read("/sys/fs/cgroup/cpuset.cpus.effective"),
        "omp": os.environ.get("OMP_NUM_THREADS"),
    }
    print("CPUDIAG " + json.dumps(line), flush=True)
