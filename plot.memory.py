# grep "@profile" mylog.txt | awk '{print $2}' > memorylist.txt


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Configuration
NUM_LISTS = 5
FILE_PATH = '/Users/bernd/python/XOR/memorylist.txt'
LIST_NAMES = [f'List {i+1}' for i in range(NUM_LISTS)]

# Initialize lists for storing separated memory data
memory_lists = [[] for _ in range(NUM_LISTS)]

# Read and parse the memorylist.txt file
with open(FILE_PATH, 'r') as file:
    lines = file.readlines()

# Skip the first line (assuming it's a comment or header)
data_lines = lines[1:]

# Process data in chunks of five lines each
for i in range(0, len(data_lines), NUM_LISTS):
    chunk = data_lines[i:i+NUM_LISTS]
    if len(chunk) < NUM_LISTS:
        print(f"Skipping incomplete chunk at lines {i+2} to {i+1+len(chunk)}")
        break  # Exit loop if chunk is incomplete
    try:
        # Convert lines to float values, stripping whitespace
        mem_values = [float(line.strip()) for line in chunk]
    except ValueError as e:
        print(
            f"Error converting lines {i+2} to {i+1+NUM_LISTS} to floats: {e}")
        break  # Exit loop on error

    # Sort the memory values numerically (ascending)
    sorted_mem = sorted(mem_values)

    # Assign sorted values to respective lists
    for list_index in range(NUM_LISTS):
        memory_lists[list_index].append(sorted_mem[list_index])

# Optional: Print the separated lists for verification
for idx, mem_list in enumerate(memory_lists):
    print(f"{LIST_NAMES[idx]}: {mem_list}")

# Determine the maximum length among the lists
max_length = max(len(lst) for lst in memory_lists)
time_axis = np.arange(1, max_length + 1)  # Convert to NumPy array

# Initialize a dictionary to store slopes
slopes = {}

# Plot the separate memory lists and calculate slopes
plt.figure(figsize=(12, 6))

for i, mem_list in enumerate(memory_lists):
    # To align all lists on the same time axis, pad shorter lists with np.nan
    padded_mem = mem_list + [np.nan]*(max_length - len(mem_list))
    plt.plot(time_axis, padded_mem, marker='o', label=LIST_NAMES[i])

    # Calculate slope using linear regression
    # Remove NaN values for accurate regression
    mem_array = np.array(padded_mem)
    valid_indices = ~np.isnan(mem_array)
    x = time_axis[valid_indices]
    y = mem_array[valid_indices]

    if len(x) < 2:
        slope = np.nan  # Not enough data points to calculate slope
    else:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

    slopes[LIST_NAMES[i]] = slope
    if not np.isnan(slope):
        print(f"Slope for {LIST_NAMES[i]}: {slope:.4f} MB per Measurement")
    else:
        print(f"Slope for {LIST_NAMES[i]}: Not enough data points")

    # Optionally, plot the regression line
    if not np.isnan(slope):
        regression_line = intercept + slope * x
        plt.plot(x, regression_line, linestyle='--',
                 label=f'{LIST_NAMES[i]} Trend')

# Customize the plot
plt.xlabel('Measurement Number')
plt.ylabel('Memory Usage (MB)')
plt.title('Memory Consumption Over Time for Separate Lists')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot to a file
plt.savefig('memory_consumption_plot_with_slopes.png')
print("Plot saved as 'memory_consumption_plot_with_slopes.png'.")

# Display the plot
plt.show()
