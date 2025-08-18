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

times_with_preemption, count_with_preemption = parse_output("results-aie-with-preemption.txt")
times_without_preemption, count_without_preemption = parse_output("results-aie-without-preemption.txt")

mean_time_with_preemption = np.mean(times_with_preemption)
mean_time_without_preemption = np.mean(times_without_preemption)
mean_count_with_preemption = np.mean(count_with_preemption)
mean_count_without_preemption = np.mean(count_without_preemption)
print(f"Number of measurements:                           {len(times_without_preemption):6} / {len(times_with_preemption):6}")
print(f"Average kernel execution time without preemption: {mean_time_without_preemption:6.0f} μs")
print(f"Average kernel execution time with preemption:    {mean_time_with_preemption:6.0f} μs")
print(f"Difference:                                       {mean_time_with_preemption - mean_time_without_preemption:6.0f} μs")
print(f"Number of layer preemptions without preempt op:   {mean_count_without_preemption:6.0f}")
print(f"Number of layer preemptions with preempt op:      {mean_count_with_preemption:6.0f}")
print(f"Difference:                                       {mean_count_with_preemption - mean_count_without_preemption:6.0f}")

fig, ax = plt.subplots()
ax.bar(0, mean_time_without_preemption, label="Without Preemption")
ax.boxplot(positions=[0], x=times_without_preemption, medianprops={"color":"black"})
ax.bar(1, mean_time_with_preemption, label="With Preemption")
ax.boxplot(positions=[1], x=times_with_preemption, medianprops={"color":"black"})
ax.set_ylabel("Kernel Execution Time (μs)")
ax.set_title("Kernel Execution Time With and Without Preemption")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Without Preemption", "With Preemption"])
plt.savefig("eval.png")
