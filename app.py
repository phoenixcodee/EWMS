# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Waste Collection Planner", layout="wide")

st.title("Waste Collecting Vehicle — KMeans + Dijkstra")
st.markdown(
    """
Upload your house coordinates and waste sensor data, or use generated sample data.
- Houses CSV format: `House_ID, X, Y`
- Waste CSV format: `House_ID, Waste_Status` (1 = waste present, 0 = no waste)
Or a single CSV with `House_ID, X, Y, Waste_Status`.
"""
)

# --------------------------
# Input: file upload or sample
# --------------------------
col1, col2 = st.columns(2)

with col1:
    houses_file = st.file_uploader("Upload houses CSV (House_ID,X,Y) or combined CSV", type=["csv"])
with col2:
    waste_file = st.file_uploader("Upload waste status CSV (House_ID,Waste_Status) — optional", type=["csv"])

use_sample = st.checkbox("Use sample random data (50 houses)", value=True if not houses_file else False)

if use_sample:
    np.random.seed(42)
    n = 50
    houses = pd.DataFrame({
        "House_ID": [f"H{i+1}" for i in range(n)],
        "X": np.random.randint(1, 100, size=n),
        "Y": np.random.randint(1, 100, size=n),
    })
    # Simulate some waste statuses
    waste_status = pd.DataFrame({
        "House_ID": houses["House_ID"],
        "Waste_Status": np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    })
else:
    # read uploaded CSV(s)
    houses = None
    if houses_file:
        df = pd.read_csv(houses_file)
        # try to detect columns
        expected = set(['House_ID', 'X', 'Y', 'Waste_Status'])
        cols = set(df.columns)
        if {'House_ID', 'X', 'Y'}.issubset(cols):
            houses = df[['House_ID', 'X', 'Y']].copy()
            if 'Waste_Status' in cols:
                waste_status = df[['House_ID', 'Waste_Status']].copy()
            else:
                waste_status = None
        else:
            st.error("Uploaded houses CSV must contain at least columns: House_ID, X, Y")
            st.stop()
    if waste_file:
        waste_status = pd.read_csv(waste_file)
        if 'House_ID' not in waste_status.columns or 'Waste_Status' not in waste_status.columns:
            st.error("Waste CSV must contain columns: House_ID, Waste_Status")
            st.stop()
    if houses is None:
        st.error("Please upload a houses CSV or check 'Use sample random data'.")
        st.stop()

# Merge to single df
df = houses.merge(waste_status, on="House_ID", how="left")
df['Waste_Status'] = df['Waste_Status'].fillna(0).astype(int)

st.subheader("Houses (first 10 rows)")
st.dataframe(df.head(10))

# --------------------------
# Parameters
# --------------------------
st.sidebar.header("Parameters")
k_clusters = st.sidebar.number_input("K for KMeans (0 = skip)", min_value=0, max_value=20, value=3, step=1)
start_x = st.sidebar.number_input("Start X coordinate", value=50)
start_y = st.sidebar.number_input("Start Y coordinate", value=50)
start_point = np.array([start_x, start_y])
run_button = st.sidebar.button("Run Planner")

# --------------------------
# Helper: build graph and compute nearest-next route using Dijkstra distances
# --------------------------
def build_complete_graph(coords, start_point=None):
    """
    coords: list of (node_id, x, y)
    start_point: (x,y) or None -> node id 'Start'
    returns networkx Graph and pos dict
    """
    G = nx.Graph()
    pos = {}
    # add house nodes
    for hid, x, y in coords:
        G.add_node(hid)
        pos[hid] = (x, y)
    # add edges between all houses
    n = len(coords)
    for i in range(n):
        hid_i, xi, yi = coords[i]
        for j in range(i+1, n):
            hid_j, xj, yj = coords[j]
            dist = np.linalg.norm(np.array([xi, yi]) - np.array([xj, yj]))
            G.add_edge(hid_i, hid_j, weight=dist)
    # add start
    if start_point is not None:
        G.add_node("Start")
        pos["Start"] = (start_point[0], start_point[1])
        for hid, x, y in coords:
            dist = np.linalg.norm(np.array([x, y]) - start_point)
            G.add_edge("Start", hid, weight=dist)
    return G, pos

def compute_greedy_route(G, waste_nodes, start="Start"):
    """
    Greedy route: from current -> nearest waste node by Dijkstra distance (not straight-line),
    visit it, remove, repeat. Finally return to start.
    Returns route list and total_distance.
    """
    route = [start]
    current = start
    total = 0.0
    remaining = waste_nodes.copy()
    while remaining:
        # compute dijkstra distances from current to each remaining
        best_node = None
        best_dist = float('inf')
        for t in remaining:
            try:
                d = nx.dijkstra_path_length(G, current, t, weight='weight')
            except nx.NetworkXNoPath:
                d = float('inf')
            if d < best_dist:
                best_dist = d
                best_node = t
        if best_node is None or best_dist == float('inf'):
            # unreachable nodes: break
            break
        total += best_dist
        # append actual shortest path nodes (optional: only append target node to route)
        # We'll append only the target node to keep route concise.
        route.append(best_node)
        current = best_node
        remaining.remove(best_node)
    # return to start
    try:
        back = nx.dijkstra_path_length(G, current, start, weight='weight')
        total += back
        route.append(start)
    except Exception:
        pass
    return route, total

# --------------------------
# Run planner
# --------------------------
if run_button:
    st.info("Running planner...")
    # Optionally do KMeans
    if k_clusters > 0:
        coords = df[['X', 'Y']].values
        k = min(k_clusters, len(df))
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(coords)
        df['Cluster'] = labels
        st.success(f"KMeans applied with k={k}")
        st.write(df['Cluster'].value_counts().sort_index())
    else:
        df['Cluster'] = -1

    # Filter waste houses only
    waste_df = df[df['Waste_Status'] == 1].reset_index(drop=True)
    if waste_df.empty:
        st.warning("No houses with waste found (Waste_Status == 1). Nothing to plan.")
    else:
        st.subheader("Waste Houses")
        st.dataframe(waste_df)

        # build graph using only houses (we still add all houses to graph to keep connectivity)
        coords_all = [(row['House_ID'], float(row['X']), float(row['Y'])) for _, row in df.iterrows()]
        G, pos = build_complete_graph(coords_all, start_point=start_point)

        # waste nodes ids
        waste_nodes = [row['House_ID'] for _, row in waste_df.iterrows()]

        route, total_distance = compute_greedy_route(G, waste_nodes, start="Start")

        st.subheader("Planned Route")
        st.write(" → ".join(route))
        st.metric("Total Distance (approx)", f"{total_distance:.2f}")

        # Prepare route dataframe for download
        route_df = pd.DataFrame({"Step": list(range(len(route))), "Node": route})
        csv_bytes = route_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download route CSV", csv_bytes, file_name="route.csv", mime="text/csv")

        # Plot graph with matplotlib
        fig, ax = plt.subplots(figsize=(7, 7))
        # draw all nodes (houses)
        xs = [pos[n][0] for n in pos if n != "Start"]
        ys = [pos[n][1] for n in pos if n != "Start"]
        ax.scatter(xs, ys, s=30, alpha=0.6)
        # highlight waste houses
        waste_xs = [float(df[df['House_ID'] == nid]['X']) for nid in waste_nodes]
        waste_ys = [float(df[df['House_ID'] == nid]['Y']) for nid in waste_nodes]
        ax.scatter(waste_xs, waste_ys, s=90, marker='s')
        # start point
        ax.scatter([start_point[0]], [start_point[1]], s=120, marker='*')

        # draw route lines (straight segments between route nodes)
        route_positions = [pos[n] for n in route if n in pos]
        rx = [p[0] for p in route_positions]
        ry = [p[1] for p in route_positions]
        ax.plot(rx, ry, linewidth=2)

        # annotate some nodes
        for n in pos:
            ax.annotate(n, xy=pos[n], xytext=(3, 3), textcoords='offset points', fontsize=8)

        ax.set_title("Waste Collection Route")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal', adjustable='box')
        st.pyplot(fig)

st.markdown("---")
st.caption("If you want a Flask or plain Python (non-Streamlit) version, say 'Flask version' or 'plain script'.")
