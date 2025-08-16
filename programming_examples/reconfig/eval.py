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

separate_xclbins = parse_output("run_0.txt")

runlist = parse_output("run_1.txt")

fused_txns = parse_output("run_2.txt")

import numpy as np

import matplotlib.pyplot as plt

bars = [
    {
        "label": "Separate XCLBins",
        "values": separate_xclbins
    },
    {
        "label": "Runlist",
        "values": runlist
    },
    {
        "label": "Fused Transactions",
        "values": fused_txns
    }
]

xs = list(range(len(bars)))

mean_separate = np.mean(separate_xclbins)
mean_runlist = np.mean(runlist)
mean_fused = np.mean(fused_txns)

print(f"Separate XCLBins:   {mean_separate:6.0f} μs"),
print(f"Runlist:            {mean_runlist:6.0f} μs")
print(f"Fused Transactions: {mean_fused:6.0f} μs")
print()
print(f"Runlist vs separate XCLBins:            {mean_runlist - mean_separate:6.0f} μs ({(mean_runlist - mean_separate) / mean_separate * 100:3.1f}%)")
print(f"Fused transactions vs runlist:          {mean_fused - mean_runlist:6.0f} μs ({(mean_fused - mean_runlist) / mean_runlist * 100:3.1f}%)")
print(f"Fused transactions vs separate XCLBins: {mean_fused - mean_separate:6.0f} μs ({(mean_fused - mean_separate) / mean_separate * 100:3.1f}%)")

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
    ax.set_title("1024x1024x1024 GEMM, followed by 1024x1024 RMS Norm (07cdf78e63)")

    plt.savefig("eval.png" if not zoomed_out else "eval_zoomed_out.png")