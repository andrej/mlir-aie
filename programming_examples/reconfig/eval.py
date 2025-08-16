#!/usr/bin/env python3

import re

def parse_output(path):
    out = []
    with open(path) as f:
        for line in f:
            # regex match the following line and extract the microseconds number as an int and add it to the list
            # "Elapsed time for all kernel executions:     6127 μs"
            match = re.search(r"Elapsed time for all kernel executions:\s+(\d+)\s+μs", line)
            if match:
                out.append(int(match.group(1)))
    return out

once = parse_output("run_0.txt")

twice_no_reconfig = parse_output("run_1.txt")

twice_with_reconfig = parse_output("run_2.txt")

import numpy as np

import matplotlib.pyplot as plt

kernel_time = np.array(twice_no_reconfig) - np.array(once)
reconfig_time = np.array(twice_with_reconfig) - np.array(twice_no_reconfig)

bars = [
    {
        "label": "Total Time",
        "values": once
    },
    {
        "label": "Kernel",
        "values": kernel_time
    },
    {
        "label": "Intra-kernel reconfig",
        "values": reconfig_time
    }
]

xs = list(range(len(bars)))

mean_once = np.mean(once)
mean_kernel = np.mean(kernel_time)
mean_reconfig = np.mean(reconfig_time)

print(f"Total time:                  {mean_once:6.0f} μs"),
print(f"Kernel time:                 {mean_kernel:6.0f} μs")
print(f"Intral-kernel reconfig time: {mean_reconfig:6.0f} μs")
print()

for zoomed_out in [True, False]:
    fig, ax = plt.subplots()
    for x, bar in zip(xs, bars):
        ax.bar(x, np.mean(bar["values"]), label=bar["label"])
        ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

    if not zoomed_out:
        ax.set_ylim([0.9*min(np.min(bar["values"]) for bar in bars), 1.01*max(np.max(bar["values"]) for bar in bars)])

    ax.set_xticks(xs)
    ax.set_xticklabels([bar["label"] for bar in bars])
    ax.set_ylabel("Total runtime (μs)")
    ax.set_title("1024x1024x1024 GEMM")

    plt.savefig("eval.png" if not zoomed_out else "eval_zoomed_out.png")