import h5py
import sumolib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configurations ---
HDF5_PATH = "od_dataset.h5"
NET_PATH = "/home/mci/mehdi/Traffic/sumo/prune_tab.net.xml"
SAMPLE_IDX = 8822  # Which simulation sample to visualize
OUTPUT_HTML = "traffic_explorer.html"

def build_interactive_dashboard():
    print(f"Loading data for sample {SAMPLE_IDX}...")
    
    # 1. Load Data
    with h5py.File(HDF5_PATH, 'r') as f:
        od_matrix = f['od_matrices'][SAMPLE_IDX]
        edge_flow = f['edge_flow'][SAMPLE_IDX]
        
    print("Parsing SUMO network for graph coordinates...")
    net = sumolib.net.readNet(NET_PATH)
    edge_id_to_idx = {edge.getID(): i for i, edge in enumerate(net.getEdges())}
    
    # 2. Prepare Graph Data (Nodes = SUMO Edges)
    node_x, node_y, node_text, node_color = [], [], [], []
    
    for sumo_edge in net.getEdges():
        idx = edge_id_to_idx[sumo_edge.getID()]
        
        # Get the physical center of the road segment to place our GNN node
        shape = sumo_edge.getShape()
        center_x = sum([p[0] for p in shape]) / len(shape)
        center_y = sum([p[1] for p in shape]) / len(shape)
        
        node_x.append(center_x)
        node_y.append(center_y)
        
        flow = edge_flow[idx]
        node_color.append(flow)
        node_text.append(f"<b>Edge ID:</b> {sumo_edge.getID()}<br><b>Flow:</b> {flow:.1f} veh")

    # Prepare Graph Connections (Lines between connected road segments)
    edge_trace_x, edge_trace_y = [], []
    for sumo_edge in net.getEdges():
        u_idx = edge_id_to_idx[sumo_edge.getID()]
        for down_edge in sumo_edge.getOutgoing():
            v_idx = edge_id_to_idx.get(down_edge.getID())
            if v_idx is not None:
                edge_trace_x.extend([node_x[u_idx], node_x[v_idx], None])
                edge_trace_y.extend([node_y[u_idx], node_y[v_idx], None])

    # 3. Build the Plotly Dashboard layout
    print("Generating Plotly visual...")
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "heatmap"}, {"type": "histogram"}],
               [{"type": "scatter", "colspan": 2}, None]],
        subplot_titles=(
            f"OD Matrix Demand (36x36)", 
            f"Edge Flow Distribution", 
            f"GNN Structure: Road Network (Nodes colored by Flow)"
        ),
        vertical_spacing=0.15
    )

    # --- Top Left: OD Matrix Heatmap ---
    fig.add_trace(
        go.Heatmap(
            z=od_matrix, 
            colorscale="Magma", 
            hovertemplate="Origin: %{y}<br>Dest: %{x}<br>Trips: %{z}<extra></extra>",
            coloraxis="coloraxis1"
        ),
        row=1, col=1
    )

    # --- Top Right: Edge Flow Histogram ---
    # Filter out absolute zeros to see the distribution of active roads better
    active_flows = edge_flow[edge_flow > 0]
    fig.add_trace(
        go.Histogram(
            x=active_flows, 
            nbinsx=50, 
            marker_color="royalblue",
            hovertemplate="Flow Range: %{x}<br>Edge Count: %{y}<extra></extra>"
        ),
        row=1, col=2
    )

    # --- Bottom: Network Graph ---
    # Add Connections (Grey lines)
    fig.add_trace(
        go.Scatter(
            x=edge_trace_x, y=edge_trace_y, 
            mode='lines', 
            line=dict(color='#cccccc', width=0.5), 
            hoverinfo='none', 
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add Nodes (Scatter points colored by flow)
    fig.add_trace(
        go.Scatter(
            x=node_x, y=node_y, 
            mode='markers', 
            hovertext=node_text, 
            hoverinfo='text', 
            marker=dict(
                size=6, 
                color=node_color, 
                colorscale='Viridis', 
                showscale=True,
                colorbar=dict(title="Flow (veh)", x=1.0, y=0.22, len=0.45)
            ), 
            showlegend=False
        ),
        row=2, col=1
    )

    # 4. Final Layout Adjustments
    fig.update_layout(
        height=1000, 
        width=1200, 
        title_text=f"Traffic Dataset Explorer | Sample: {SAMPLE_IDX}",
        title_font_size=20,
        coloraxis1=dict(colorscale="Magma", colorbar=dict(title="Trips", x=0.45, y=0.8, len=0.35)),
        template="plotly_white",
        hovermode="closest"
    )
    
    # Update axes titles
    fig.update_xaxes(title_text="Destination Zone", row=1, col=1)
    fig.update_yaxes(title_text="Origin Zone", row=1, col=1)
    fig.update_xaxes(title_text="Flow Volume", row=1, col=2)
    fig.update_yaxes(title_text="Number of Edges", row=1, col=2)
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=2, col=1) # Keep map proportions correct

    # Save to file
    fig.write_html(OUTPUT_HTML)
    print(f"Success! Dashboard saved to {OUTPUT_HTML}.")
    print("Open this file in your web browser to interact with the data.")

if __name__ == "__main__":
    build_interactive_dashboard()