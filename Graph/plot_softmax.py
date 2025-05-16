import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.colors as mcolors

N_values = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
nbr_datas = ["Kernel Execution Time", " Memory Allocation Time"]
method_names = ['naive', 'online', 'sharedmem', 'shfl', 'vectorized', 'blocktiling']
color_palette = plt.get_cmap("tab10")
colors = [color_palette(i) for i in range(len(method_names))]
titre = ['softmax benchmark of kernel time', 'softmax benchmark for Memory allocation time']

methods = [[[[] for _ in range(len(nbr_datas))] for _ in range(len(N_values))] for _ in range(len(method_names))]
filtered_averages = [[[] for _ in range(len(nbr_datas))] for _ in range(len(method_names))]
filtered_points = [[[[] for _ in range(len(nbr_datas))] for _ in range(len(N_values))] for _ in range(len(method_names))]

with open('data_softmax_computer.txt', 'r') as file:
    lines = file.readlines()

counter_line = 0
for i in range(len(method_names)):
    for j in range(len(nbr_datas)):
        for k in range(len(N_values)):
            values = list(map(float, lines[counter_line].strip().split()))
            methods[i][k][j].extend(values)
            counter_line += 1

def filter_and_recalculate(data):
    if len(data) <= 2:
        return None, None
    mu, sigma = np.mean(data), np.std(data)
    lower_bound = norm.ppf(0.025, loc=mu, scale=sigma)
    upper_bound = norm.ppf(0.975, loc=mu, scale=sigma)
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data, np.mean(filtered_data)

for i in range(len(methods)):
    for j in range(len(methods[i])):
        kernel_data = methods[i][j][0]
        device_data = methods[i][j][1]
        filtered_kernel, avg_kernel = filter_and_recalculate(kernel_data)
        filtered_device, avg_device = filter_and_recalculate(device_data)
        if avg_kernel is not None:
            filtered_averages[i][0].append(avg_kernel)
            filtered_points[i][j][0] = filtered_kernel
        if avg_device is not None:
            filtered_averages[i][1].append(avg_device)
            filtered_points[i][j][1] = filtered_device

for i in range(len(nbr_datas)):
    plt.figure(figsize=(12, 6))
    for j in range(len(filtered_averages)):
        plt.plot(N_values, filtered_averages[j][i], label=method_names[j],
                 color=colors[j], marker='s', linewidth=2)
        for k in range(len(filtered_points[j])):
            for l in range(len(filtered_points[j][k][i])):
                plt.scatter(N_values[k], filtered_points[j][k][i][l],
                            color=colors[j], marker='x')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(N_values)
    plt.gca().set_xticklabels(N_values)
    plt.xlabel('size of N (log scale)')
    plt.ylabel(nbr_datas[i] + ' [ms] (log scale)')
    plt.title(titre[i])
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

# Barplot empilé avec hachures
kernel_colors = [mcolors.to_rgba(c, alpha=0.5) for c in colors]
device_colors = [mcolors.to_rgba(c, alpha=1.0) for c in colors]
hatch_pattern = '//'  # motif de hachure

log_N_values = np.log10(N_values)
bar_group_width = 0.18
bar_width = bar_group_width / len(method_names)

plt.figure(figsize=(12, 6))

for i in range(len(method_names)):
    positions = log_N_values + (i - 2.5) * bar_width
    device_vals = filtered_averages[i][1]
    kernel_vals = filtered_averages[i][0]
    plt.bar(positions, device_vals, width=bar_width,
            color=device_colors[i], label=f"{method_names[i]} - Device")
    plt.bar(positions, kernel_vals, width=bar_width, bottom=device_vals,
            color=kernel_colors[i], hatch=hatch_pattern, edgecolor=device_colors[i],
            label=f"{method_names[i]} - Kernel")

plt.xticks(log_N_values, N_values)
plt.xlabel("Size of N (log scale)")
plt.ylabel("Total Time [ms]")
plt.title("softmax total time (kernel + Memory)")
plt.legend(title="Method & Component")
plt.tight_layout()
plt.show()