#!/usr/bin/env python3
import matplotlib.pyplot as plt


compute_times = [
        (3135,   {"label": "Preamble/Postamble"}), 
        (448,    {"label": "Zeroing Output"}), 
        (6285,   {"label": "Unknown"}),
        (148960, {"label": "Compute Mat. Mul."})
]

data_movement_times = [
        (117950, {"label": "Data Movement"})
]

theoretical_min = [
        (134217, {"label": "Theoretical Minimum @ 128 GOP/s"})
]

def stacked_barh(y, data, **kwargs):
    left = 0
    for x, data_point_args in data:
        args = {**kwargs, **data_point_args} 
        plt.barh(y, x, left=left, **args)
        left += x

plt.style.use('dark_background')
fig = plt.gcf()
fig.set_size_inches(6.4, 3.5, forward=True)

stacked_barh(1, compute_times)
stacked_barh(0, data_movement_times)
stacked_barh(-1, theoretical_min)

plt.legend(loc='upper left', bbox_to_anchor=(0, -0.3), ncols=2)
plt.xlabel("Time [microseconds]")
plt.yticks([])
plt.title("Matrix Multiplication Execution Time (2048x2048x2048 i16)")
plt.tight_layout()
plt.show()
