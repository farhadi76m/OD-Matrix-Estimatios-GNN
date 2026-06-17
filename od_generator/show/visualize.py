import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your generated dataset
HDF5_PATH = "od_dataset.h5"

def load_and_visualize_sample(sample_idx=0):
    # 1. Load the data
    print(f"Loading sample {sample_idx} from {HDF5_PATH}...")
    with h5py.File(HDF5_PATH, 'r') as f:
        # Extract OD Matrix (Shape: 36x36)
        od_matrix = f['od_matrices'][sample_idx]
        
        # Extract Edge Statics (Shape: 1897)
        edge_flow = f['edge_flow'][sample_idx]
        edge_speed = f['edge_speed'][sample_idx]
        edge_density = f['edge_density'][sample_idx]
        
    print(f"OD Matrix shape: {od_matrix.shape}")
    print(f"Edge Flow shape: {edge_flow.shape}")

    # 2. Setup the Visualization (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- LEFT: OD Matrix Heatmap ---
    sns.heatmap(od_matrix, ax=axes[0], cmap="magma", cbar_kws={'label': 'Number of Trips'})
    axes[0].set_title(f"OD Matrix Demand (Sample {sample_idx})\n36x36 TAZ Grid", fontsize=14)
    axes[0].set_xlabel("Destination Zone (TAZ)", fontsize=12)
    axes[0].set_ylabel("Origin Zone (TAZ)", fontsize=12)

    # --- RIGHT: Edge Statics (Distribution) ---
    # Since plotting 1897 individual edge lines is messy, a histogram showing 
    # the distribution of flow across the network is the best way to view the statics.
    sns.histplot(edge_flow[edge_flow > 0], bins=50, ax=axes[1], color="royalblue", kde=True)
    axes[1].set_title(f"Network Edge Flow Distribution (Sample {sample_idx})\nExcluding 0-flow edges", fontsize=14)
    axes[1].set_xlabel("Flow Volume (Vehicles per Edge)", fontsize=12)
    axes[1].set_ylabel("Frequency (Number of Edges)", fontsize=12)

    # Add text box with summary statistics for the edges
    stats_text = (
        f"Network Averages:\n"
        f"Mean Flow: {np.mean(edge_flow):.1f} veh\n"
        f"Max Flow: {np.max(edge_flow):.1f} veh\n"
        f"Mean Speed: {np.mean(edge_speed):.2f} m/s\n"
        f"Max Density: {np.max(edge_density):.2f} veh/km"
    )
    axes[1].text(0.65, 0.85, stats_text, transform=axes[1].transAxes, 
                 fontsize=11, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()
    plt.savefig(f"od_sample_{sample_idx}_visualization.png", dpi=300)

if __name__ == "__main__":
    load_and_visualize_sample(sample_idx=0)