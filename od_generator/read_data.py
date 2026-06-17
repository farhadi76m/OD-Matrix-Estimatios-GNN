import sumolib
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import numpy as np
import h5py
from pathlib import Path


def read_hdf5_sample(hdf5_path, sample_idx):
    """
    Read OD matrix and edge flow for a specific sample.
    
    Args:
        hdf5_path (str): Path to HDF5 file
        sample_idx (int): Sample index
        
    Returns:
        tuple: (od_matrix [36x36], edge_flow [n_edges])
    """
    with h5py.File(hdf5_path, 'r') as f:
        od_matrix = f['od_matrices'][sample_idx][:]  # Copy to memory
        edge_flow = f['edge_flow'][sample_idx][:]
        breakpoint()    
    return od_matrix, edge_flow


def parse_taz_zones(taz_file):
    """
    Parse TAZ file and extract zone info (id, shape).
    PRESERVES order → zone index = position in XML file = OD matrix row/col index
    
    Returns:
        dict: {zone_id: {"shape": [(x,y), ...], "xs": [...], "ys": [...], "idx": int}}
        list: [zone_id, ...] in XML order (for index mapping)
    """
    zones = {}
    zone_order = []  # Preserve XML order
    tree = ET.parse(taz_file)
    root = tree.getroot()
    
    for idx, taz in enumerate(root.findall("taz")):
        zone_id = taz.attrib["id"]
        zone_order.append(zone_id)
        
        # Parse shape coordinates
        coords = [
            tuple(map(float, p.split(",")))
            for p in taz.attrib["shape"].split()
        ]
        
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        
        zones[zone_id] = {
            "shape": coords,
            "xs": xs,
            "ys": ys,
            "idx": idx,  # OD matrix index
        }
    
    return zones, zone_order


def od_to_zone_demand(od_matrix, zone_order, agg_mode="total"):
    """
    Convert 36x36 OD matrix to per-zone demand using zone ordering.
    
    Args:
        od_matrix: (36, 36) array where [i,j] = trips from zone i to zone j
        zone_order: list of zone_ids in XML/OD matrix order
        agg_mode: 
            - "origin": sum over destinations → outgoing demand per zone
            - "dest":   sum over origins → incoming demand per zone
            - "total":  origin + dest → total demand per zone
    
    Returns:
        dict: {zone_id: demand_value}
    """
    origin_demand = od_matrix.sum(axis=1)  # Sum over columns (destinations)
    dest_demand = od_matrix.sum(axis=0)     # Sum over rows (origins)
    
    if agg_mode == "origin":
        zone_demand = origin_demand
    elif agg_mode == "dest":
        zone_demand = dest_demand
    elif agg_mode == "total":
        zone_demand = origin_demand + dest_demand
    else:
        raise ValueError(f"Unknown agg_mode: {agg_mode}")
    
    # Map to zone IDs using zone_order (matches OD matrix indices)
    demand_dict = {zone_order[i]: float(zone_demand[i]) for i in range(len(zone_order))}
    
    return demand_dict


def normalize_color(value, min_val, max_val):
    """Map value to [0, 1] for color intensity."""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def visualize_od_on_taz(sample_idx=0, hdf5_path="od_dataset.h5", 
                        taz_file="../sumo/tehran_taz.xml", agg_mode="total"):
    """
    Load OD matrix, aggregate to zones, and visualize on TAZ map.
    
    Args:
        sample_idx (int): Which sample to visualize
        hdf5_path (str): Path to HDF5 dataset
        taz_file (str): Path to TAZ XML file
        agg_mode (str): "origin", "dest", or "total"
    """
    
    # Load data
    print(f"Loading sample {sample_idx} from {hdf5_path}...")
    od_matrix, edge_flow = read_hdf5_sample(hdf5_path, sample_idx)
    
    print(f"OD matrix shape: {od_matrix.shape}")
    print(f"Total demand: {od_matrix.sum():.0f} trips")
    
    # Parse zones
    print(f"Parsing zones from {taz_file}...")
    zones, zone_order = parse_taz_zones(taz_file)
    print(f"Found {len(zones)} zones")
    
    # Aggregate OD to zone demand (now with correct mapping)
    demand_dict = od_to_zone_demand(od_matrix, zone_order, agg_mode=agg_mode)
    
    # Create figure
    fig = go.Figure()
    
    # Draw TAZ zones colored by demand FIRST (as background)
    print("Drawing TAZ zones...")
    demand_values = list(demand_dict.values())
    min_demand = min(demand_values)
    max_demand = max(demand_values)
    
    for zone_id, zone_info in sorted(zones.items()):
        xs = zone_info["xs"]
        ys = zone_info["ys"]
        
        demand = demand_dict.get(zone_id, 0)
        norm_demand = normalize_color(demand, min_demand, max_demand)
        
        # Red intensity based on demand (lower opacity for background)
        r = int(255 * norm_demand)
        g = int(100 * (1 - norm_demand))
        b = int(100 * (1 - norm_demand))
        
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                fill="toself",
                mode="lines",
                fillcolor=f"rgba({r},{g},{b},0.5)",
                line=dict(color="darkred", width=0.5),
                name=zone_id,
                hovertemplate=(
                    f"<b>Zone {zone_id}</b><br>"
                    f"Demand ({agg_mode}): {demand:.0f} trips<br>"
                    f"<extra></extra>"
                )
            )
        )
    
    # Draw network edges AFTER zones (so they appear on top)
    print("Drawing network edges...")
    net = sumolib.net.readNet("../sumo/prune_tab.net.xml")
    for edge in net.getEdges():
        shape = edge.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color="rgba(100,100,100,0.4)", width=0.8),
                hoverinfo="skip",
                showlegend=False,
                name="network"
            )
        )
    
    # Layout
    fig.update_layout(
        title=f"OD Demand Map (Sample {sample_idx}, mode={agg_mode})<br>"
              f"Total Demand: {od_matrix.sum():.0f} trips",
        hovermode="closest",
        showlegend=False,
        width=1200,
        height=900,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    fig.show()
    
    # Print stats
    print(f"\n{'Zone':<8} {'Demand':<10} {'%':<6}")
    print("-" * 24)
    for zone_id in sorted(zones.keys(), key=lambda z: demand_dict.get(z, 0), reverse=True):
        d = demand_dict.get(zone_id, 0)
        pct = 100 * d / od_matrix.sum()
        print(f"{zone_id:<8} {d:<10.0f} {pct:<6.1f}%")


if __name__ == "__main__":
    visualize_od_on_taz(
        sample_idx=3124,
        hdf5_path="od_dataset.h5",
        taz_file="../sumo/tehran_taz.xml",
        agg_mode="total"  # try "origin", "dest", or "total"
    )