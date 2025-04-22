import matplotlib.pyplot as plt
import numpy as np

data1 = {
    'LightGCN': [0.0126, 0.0124, 0.0122, 0.0120],
    'MACR_LGN': [0.0255, 0.0251, 0.0245, 0.0244],
    'DICE_LGN': [0.0371, 0.0370, 0.0358, 0.0356],
    'IPS_LGN': [0.0252, 0.0250, 0.0246, 0.0240],
    'DCCL_LGN': [0.0238, 0.0237, 0.0231, 0.0228],
    'PCDR_LGN': [0.0489, 0.0485, 0.0476, 0.0476]
}

data2 = {
    'LightGCN': [0.0199, 0.0197, 0.0193, 0.0191],
    'MACR_LGN': [0.0337, 0.0332, 0.0325, 0.0322],
    'DICE_LGN': [0.0445, 0.0446, 0.0430, 0.0420],
    'IPS_LGN': [0.0333, 0.0332, 0.0324, 0.0316],
    'DCCL_LGN': [0.0310, 0.0303, 0.0299, 0.0294],
    'PCDR_LGN': [0.0497, 0.0504, 0.0526, 0.0497]
}

data3 = {
    'LightGCN': [0.2101, 0.2075, 0.2044, 0.2012],
    'MACR_LGN': [0.3508, 0.3477, 0.3430, 0.3445],
    'DICE_LGN': [0.4398, 0.4418, 0.4335, 0.4357],
    'IPS_LGN': [0.3472, 0.3498, 0.3455, 0.3402],
    'DCCL_LGN': [0.3210, 0.3225, 0.3138, 0.3140],
    'PCDR_LGN': [0.4528, 0.4508, 0.4556, 0.4445]
}

# X-axis values
x_values = np.arange(0, 0.15, 0.04)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(10, 3))  # 1 row, 3 columns of subplots

# Define colors and markers for each line
colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
markers = ['o', 's', '^', 'v', 'D', 'P', '*']

# Function to plot data on a given subplot
def plot_data(ax, data, y_label):
    i = 0
    for method, y_values in data.items():
        ax.plot(x_values, y_values, label=method, color=colors[i], marker=markers[i], linestyle='-', linewidth=1, markersize=5)
        i += 1
    ax.set_xlabel('noise ratio', fontsize=12)  # X-axis label
    ax.set_ylabel(y_label, fontsize=12)  # Y-axis label
    ax.set_xticks(x_values)  # X-axis ticks
    ax.tick_params(axis='x', labelsize=10)  # X-axis ticks fontsize
    ax.tick_params(axis='y', labelsize=8)  # Y-axis ticks fontsize
    ax.grid(True, linestyle='--', alpha=0.2)  # Add a grid for easier reading
    # ax.legend(loc='best', fontsize=7, frameon=False)  # Show legend, remove frame



# Plot data on each subplot
plot_data(axes[0], data1, 'Recall@10')
plot_data(axes[1], data2, 'NDCG@10')
plot_data(axes[2], data3, 'HR@10')


# Create a single legend for all subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), fontsize=10, frameon=True)

# Adjust margins and spacing
plt.tight_layout(rect=[0, 0.1, 1, 0.88])  # Adjust layout to make space for the legend


# Save the plot (optional)
plt.savefig('line_plot_subplots.svg', dpi=300)  # Save as a high-resolution image

# Show the plot
plt.show()
