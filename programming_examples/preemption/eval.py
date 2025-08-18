#!/usr/bin/env python3

import numpy
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_output(path):
    times = []
    preemptions = []
    with open(path) as f:
        for line in f:
            match_time = re.search(r"Elapsed time for kernel execution:\s+(\d+)\s+μs", line)
            if match_time:
                times.append(int(match_time.group(1)))
            match_preemption = re.search(r"  \|15      \|0       \|(\d+)\s+|6             \|", line)
            if match_preemption:
                preemptions.append(int(match_preemption.group(1)))
    assert len(times) == len(preemptions)
    return times, preemptions

times_with_preemption, count_with_preemption = parse_output("results-aie-with-preemption-enable.txt")
times_without_preemption, count_without_preemption = parse_output("results-aie-without-preemption-enable.txt")
times_with_preemption_disabled, count_with_preemption_disabled = parse_output("results-aie-with-preemption-disable.txt")
times_without_preemption_disabled, count_without_preemption_disabled = parse_output("results-aie-without-preemption-disable.txt")

mean_time_with_preemption = np.mean(times_with_preemption)
mean_time_without_preemption = np.mean(times_without_preemption)
mean_count_with_preemption = np.mean(count_with_preemption)
mean_count_without_preemption = np.mean(count_without_preemption)
mean_time_with_preemption_disabled = np.mean(times_with_preemption_disabled)
mean_time_without_preemption_disabled = np.mean(times_without_preemption_disabled)
mean_count_with_preemption_disabled = np.mean(count_with_preemption_disabled)
mean_count_without_preemption_disabled = np.mean(count_without_preemption_disabled)
print(f"With forced preemption enabled:")
print(f"----")
print(f"Number of measurements:                           {len(times_without_preemption):6} / {len(times_with_preemption):6}")
print(f"Average kernel execution time without preemption: {mean_time_without_preemption:6.0f} μs")
print(f"Average kernel execution time with preemption:    {mean_time_with_preemption:6.0f} μs")
print(f"Difference:                                       {mean_time_with_preemption - mean_time_without_preemption:6.0f} μs")
print(f"Number of layer preemptions without preempt op:   {mean_count_without_preemption:6.0f}")
print(f"Number of layer preemptions with preempt op:      {mean_count_with_preemption:6.0f}")
print(f"Difference:                                       {mean_count_with_preemption - mean_count_without_preemption:6.0f}")
print()
print(f"With forced preemption disabled:")
print(f"----")
print(f"Number of measurements:                           {len(times_without_preemption_disabled):6} / {len(times_with_preemption_disabled):6}")
print(f"Average kernel execution time without preemption: {mean_time_without_preemption_disabled:6.0f} μs")
print(f"Average kernel execution time with preemption:    {mean_time_with_preemption_disabled:6.0f} μs")
print(f"Difference:                                       {mean_time_with_preemption_disabled - mean_time_without_preemption_disabled:6.0f} μs")
print(f"Number of layer preemptions without preempt op:   {mean_count_without_preemption_disabled:6.0f}")
print(f"Number of layer preemptions with preempt op:      {mean_count_with_preemption_disabled:6.0f}")
print(f"Difference:                                       {mean_count_with_preemption_disabled - mean_count_without_preemption_disabled:6.0f}")

bars = [
    ("Without preempt op", times_without_preemption_disabled),
    ("With preempt op", times_with_preemption_disabled),
    ("Without preempt op,\nforced preemption", times_without_preemption),
    ("With preempt op,\nforced preemption", times_with_preemption)
]

fig, ax = plt.subplots()
for i, (label, values) in enumerate(bars):
    ax.bar(i, np.mean(values), label=label)
    ax.boxplot(positions=[i], x=values, medianprops={"color":"black"})
    ax.set_ylabel("Kernel Execution Time (μs)")
    ax.set_title("Kernel Execution Time With and Without Preemption")

ax.legend(ncols=2)
ax.set_xticks([])
plt.savefig("eval.png")
