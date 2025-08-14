#!/usr/bin/env python3

import re
import numpy as np
import matplotlib.pyplot as plt

def parse_memcpy_output(path):
    out = []
    with open(path) as f:
        for line in f:
            match = re.search(r"Effective Bandwidth: ([0-9\.]+) GB/s", line)
            if match:
                out.append(float(match.group(1)))
    return out

def parse_mv_output(path):
    out = []
    with open(path) as f:
        for line in f:
            match = re.search(r"Avg NPU gflops:\s*([0-9\.]+)", line)
            if match:
                out.append(float(match.group(1)))
    return out


# --------------------------------------------------------------------------
# memcpy

bars = [
    {
        "label": "Strix (4)",
        "values": parse_memcpy_output("memcpy_4col_stx.txt")
    },
    {
        "label": "Krackan (4)",
        "values": parse_memcpy_output("memcpy_4col_krk.txt")
    },
    {
        "label": "Strix (8)",
        "values": parse_memcpy_output("memcpy_8col_stx.txt") 
    },
    {
        "label": "Krackan (8)",
        "values": parse_memcpy_output("memcpy_8col_krk.txt")
    },
]
for bar in bars:
    print(f"{bar['label']}: {np.mean(bar['values']):0.2f} GB/s")

xs = list(range(len(bars)))
fig, ax = plt.subplots()
for x, bar in zip(xs, bars):
    ax.bar(x, np.mean(bar["values"]), label=bar["label"])
    ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

ax.set_xticks(xs)
ax.set_xticklabels([bar["label"] for bar in bars])
ax.set_ylabel("Memory throughput (GB/s)")
ax.set_title("memcpy benchmark (commit 10b917a9)")

plt.savefig("eval_memcpy.png")


# --------------------------------------------------------------------------
# memcpy (no bypass)

bars = [
    {
        "label": "Strix (4)",
        "values": parse_memcpy_output("memcpy_4col_nobypass_stx.txt")
    },
    {
        "label": "Krackan (4)",
        "values": parse_memcpy_output("memcpy_4col_nobypass_krk.txt")
    },
    {
        "label": "Strix (8)",
        "values": parse_memcpy_output("memcpy_8col_nobypass_stx.txt")
    },
    {
        "label": "Krackan (8)",
        "values": parse_memcpy_output("memcpy_8col_nobypass_krk.txt")
    },
]
for bar in bars:
    print(f"{bar['label']}: {np.mean(bar['values']):0.2f} GB/s")

xs = list(range(len(bars)))
fig, ax = plt.subplots()
for x, bar in zip(xs, bars):
    ax.bar(x, np.mean(bar["values"]), label=bar["label"])
    ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

ax.set_xticks(xs)
ax.set_xticklabels([bar["label"] for bar in bars])
ax.set_ylabel("Memory throughput (GB/s)")
ax.set_title("memcpy benchmark (no bypass) (commit 10b917a9)")

plt.savefig("eval_memcpy_nobypass.png")


# --------------------------------------------------------------------------
# GEMV

bars = [
    {
        "label": "Strix",
        "values": parse_mv_output("mv_4096x4096_64x64_stx.txt")
    },
    {
        "label": "Krackan",
        "values": parse_mv_output("mv_4096x4096_64x64_krk.txt")
    },
]
for bar in bars:
    print(f"{bar['label']}: {np.mean(bar['values']):0.2f} GFLOP/s")

xs = list(range(len(bars)))
plt.figure()
fig, ax = plt.subplots()
for x, bar in zip(xs, bars):
    ax.bar(x, np.mean(bar["values"]), label=bar["label"])
    ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

ax.set_xticks(xs)
ax.set_xticklabels([bar["label"] for bar in bars])
ax.set_ylabel("Compute throughput (GFLOP/s)")
ax.set_title("matrix-vector multiplication (4096x4096x1) (commit e17dcbc)")

plt.savefig("eval_mv.png")