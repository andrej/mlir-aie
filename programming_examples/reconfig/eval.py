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

no_preemption = parse_output("run_0.txt")

with_preemption = parse_output("run_1.txt")


import numpy as np

import matplotlib.pyplot as plt

bars = [
    {
        "label": "No preemption",
        "values": no_preemption
    },
    {
        "label": "With preemptoin",
        "values": with_preemption
    },
]

xs = list(range(len(bars)))

mean_no_preemption = np.mean(no_preemption)
mean_kernel = np.mean(with_preemption)

print(f"Total time w preemption:     {mean_no_preemption:6.0f} μs"),
print(f"Total time wo preemption:    {mean_kernel:6.0f} μs")
print(f"Difference:                  {mean_no_preemption-mean_kernel:6.0f} μs ({(mean_no_preemption-mean_kernel)/mean_no_preemption*100:3.1f}%)")
print()

for zoomed_out in [True, False]:
    fig, ax = plt.subplots()
    for x, bar in zip(xs, bars):
        ax.bar(x, np.mean(bar["values"]), label=bar["label"])
        ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

    if not zoomed_out:
        ax.set_ylim([0.9*min(np.min(bar["values"]) for bar in bars[0:2]), 1.01*max(np.max(bar["values"]) for bar in bars)])

    ax.set_xticks(xs)
    ax.set_xticklabels([bar["label"] for bar in bars])
    ax.set_ylabel("Total runtime (μs)")
    ax.set_title("1024x1024x1024 GEMM")

    plt.savefig("eval.png" if not zoomed_out else "eval_zoomed_out.png")