import re
import matplotlib.pyplot as plt
import numpy as np

# Regex to extract iteration and loss
pattern = re.compile(r'(\d+)/\d+.*?loss=([0-9.]+)')

# Storage for runs
runs = []
current_run = []

# Parse the file
with open('gpt2_error_6.txt', 'r') as f:
    last_iteration = -1
    for line in f:
        match = pattern.search(line)
        if match:
            iteration = int(match.group(1))
            loss = float(match.group(2))

            # Detect start of a new run
            if iteration < last_iteration:
                if current_run:
                    runs.append(current_run)
                    current_run = []
            current_run.append((iteration, loss))
            last_iteration = iteration

    # Add the last run
    if current_run:
        runs.append(current_run)

# Save and plot each run if it has >= 10,000 iterations
plot_count = 0
for idx, run in enumerate(runs):
    iterations, losses = zip(*run)

    if iterations[-1] - iterations[0] < 10_000:
        print(f"Skipping Run {idx+1}: Only {iterations[-1] - iterations[0]} iterations")
        continue

    # Normalize iteration to start from 1 (avoid log(0))
    norm_iterations = np.array([i - iterations[0] + 1 for i in iterations])
    losses = np.array(losses)

    # Save data to .npz
    np.savez(f'loss_data_run_{idx+1}.npz', iterations=norm_iterations, losses=losses)

    # Plot (log-log)
    plt.figure(figsize=(10, 6))
    plt.plot(norm_iterations, losses, label=f'Run {idx+1}', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iteration (log scale)')
    plt.ylabel('Loss (log scale)')
    plt.title(f'Training Loss - Run {idx+1} (log-log)')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig(f'loss_plot_run_{idx+1}.png')
    plt.close()

    plot_count += 1

print(f"Saved {plot_count} plots and .npz files (skipped {len(runs) - plot_count} short runs).")
