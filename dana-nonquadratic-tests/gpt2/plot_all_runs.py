import numpy as np
import matplotlib.pyplot as plt
import glob

def moving_average(data, window_size):
    """Compute moving average with given window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

# Find all npz files
npz_files = glob.glob('loss_data_run_*.npz')

if not npz_files:
    print("No .npz files found! Please run read_txt.py first.")
    exit(1)

# Create the plot
plt.figure(figsize=(12, 8))

# Window size for moving average (adjust this value to change smoothing)
window_size = 100

# Create yellow to purple colormap
num_files = len(sorted(npz_files))
colors = plt.cm.YlOrRd_r(np.linspace(0, 1, num_files))  # Yellow to purple gradient

# Plot each run with a different color
for idx, npz_file in enumerate(sorted(npz_files)):
    data = np.load(npz_file)
    iterations = data['iterations']
    losses = data['losses']
    
    # Extract run number and p value from filename
    # Assuming filename format ends with something like "p_0.1.npz"
    p_value = npz_file.split('_')[-1].split('.npz')[0]
    
    # Plot original data with low opacity
    plt.plot(iterations, losses, alpha=0.2, color=colors[idx])
    
    # Compute and plot moving average
    smooth_losses = moving_average(losses, window_size)
    # Adjust iterations array to match smoothed data length
    smooth_iterations = iterations[window_size-1:]
    plt.plot(smooth_iterations, smooth_losses, 
             label=f'p={p_value} (MA-{window_size})', 
             color=colors[idx],
             linewidth=2)

# Set up the plot
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Iteration (log scale)')
plt.ylabel('Loss (log scale)')
plt.title(f'Training Loss - All Runs (log-log)\nwith {window_size}-point Moving Average\nBatch Size: 32, Sequence Length: 1024')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# Add some padding to the axes
plt.margins(x=0.02)

# Save and show the plot
plt.savefig('combined_loss_plot_smooth.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Smoothed combined plot saved as 'combined_loss_plot_smooth.png'") 