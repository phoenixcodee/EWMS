import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Waste Collection Vehicle", layout="wide")

st.title("🚛 Waste Collecting Vehicle – K-Means + Dijkstra (No CSV Required)")

st.write("This app automatically generates 50 random houses with waste sensors.")

# ----------------------------
# Generate Random House Data
# ----------------------------
np.random.seed(42)
num_houses = 50

houses = {
    f"H{i+1}": (np.random.randint(1, 100), np.random.randint(1, 100))
    for i in range(num_houses)
}

# Waste Sensor: 1 = Waste Present, 0 = No Waste
waste_sensor = {
    hid: np.random.choice([0, 1], p=[0.6, 0.4])
    for hid in houses
}

start_point = (50, 50)

st.subheader("Sample Generated House Data (First 10)")
st.write({k: houses[k] for k in list(houses.keys())[:10]})

st.subheader("Waste Status Summary")
st.write(f"Total Waste Houses: {sum(waste_sensor.values())}")

# ----------------------------
# Run Button
# ----------------------------
if st.button("Generate Route"):
    # ----------------------------
    # K-Means clustering (optional)
    # ----------------------------
    coords_array = np.array(list(houses.values()))
    kmeans = KMeans(n_clusters=3, random_state=42).fit(coords_array)
    labels = kmeans.labels_

    # ----------------------------
    # Build Graph for Dijkstra
    # ----------------------------
    G = nx.Graph()
    positions = {}

    # Add house nodes
    for i, (hid, (x, y)) in enumerate(houses.items()):
        G.add_node(hid)
        positions[hid] = (x, y)
    
    # Add edges between all houses (complete graph)
    hid_list = list(houses.keys())
    for i in range(num_houses):
        for j in range(i + 1, num_houses):
            h1, h2 = hid_list[i], hid_list[j]
            dist = np.linalg.norm(np.array(houses[h1]) - np.array(houses[h2]))
            G.add_edge(h1, h2, weight=dist)

    # Add Start node
    G.add_node("Start")
    positions["Start"] = start_point

    for hid, (x, y) in houses.items():
        dist = np.linalg.norm(np.array([x, y]) - np.array(start_point))
        G.add_edge("Start", hid, weight=dist)

    # ----------------------------
    # Dijkstra Route Calculation
    # ----------------------------
    waste_houses = [hid for hid, status in waste_sensor.items() if status == 1]

    route = ["Start"]
    current = "Start"
    total_distance = 0

    remaining = waste_houses.copy()

    while remaining:
        shortest = {}
        for target in remaining:
            d = nx.dijkstra_path_length(G, current, target, weight='weight')
            shortest[target] = d

        next_house = min(shortest, key=shortest.get)
        total_distance += shortest[next_house]
        route.append(next_house)
        current = next_house
        remaining.remove(next_house)

    # Return to start
    total_distance += nx.dijkstra_path_length(G, current, "Start")
    route.append("Start")

    # ----------------------------
    # Show Route
    # ----------------------------
    st.subheader("📍 Final Route")
    st.write(" → ".join(route))
    st.metric("Total Distance", f"{total_distance:.2f}")

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot houses
    for hid, (x, y) in houses.items():
        color = "red" if waste_sensor[hid] == 1 else "blue"
        ax.scatter(x, y, color=color, s=50)
        ax.text(x + 1, y + 1, hid, fontsize=7)

    # Plot start
    ax.scatter(start_point[0], start_point[1], color="green", s=100, marker="*")
    ax.text(start_point[0] + 1, start_point[1] + 1, "Start", fontsize=8)

    # Route lines
    for i in range(len(route) - 1):
        x1, y1 = positions[route[i]]
        x2, y2 = positions[route[i+1]]
        ax.plot([x1, x2], [y1, y2], linewidth=2)

    ax.set_title("Waste Collection Route")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    st.pyplot(fig)

st.caption("✔ No CSV Needed. All Data Auto-Generated.")
