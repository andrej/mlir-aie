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

runlist = parse_output("run_3.txt")

fused_txns = parse_output("run_1.txt")

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

fig, ax = plt.subplots()
for x, bar in zip(xs, bars):
    ax.bar(x, np.mean(bar["values"]), label=bar["label"])
    ax.boxplot(positions=[x], x=bar["values"], medianprops={"color" : "black"})

ax.set_xticks(xs)
ax.set_xticklabels([bar["label"] for bar in bars])
ax.set_ylabel("Total runtime (μs)")
ax.set_title("Running add two, subtract three (commit ?)")

plt.savefig("eval.png")