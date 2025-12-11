import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# SAMPLE DATA
# ------------------------------------------------------------

# 50 houses with X,Y coordinates
np.random.seed(42)
houses = np.random.randint(1, 100, size=(50, 2))

# Random waste sensor values (0 means no waste, 1 means waste present)
waste_sensor = np.random.randint(0, 2, size=50)

# Starting point
start_point = np.array([50, 50])

# ------------------------------------------------------------
# STEP 1: APPLY K-MEANS CLUSTERING
# ------------------------------------------------------------

k = 3   # Number of clusters
model = KMeans(n_clusters=k, random_state=42)
model.fit(houses)

cluster_labels = model.labels_

# Filter houses with waste only
waste_houses = houses[waste_sensor == 1]

print("Total Houses:", len(houses))
print("Waste Houses:", len(waste_houses))

# ------------------------------------------------------------
# STEP 2: BUILD GRAPH FOR DIJKSTRA SHORTEST PATHS
# ------------------------------------------------------------

G = nx.Graph()

# Add all houses + starting point as nodes
for idx, h in enumerate(houses):
    G.add_node(f"H{idx}", pos=(h[0], h[1]))

G.add_node("Start", pos=(start_point[0], start_point[1]))

# Add full connections (complete graph) with distances
for i in range(len(houses)):
    for j in range(i + 1, len(houses)):
        dist = np.linalg.norm(houses[i] - houses[j])
        G.add_edge(f"H{i}", f"H{j}", weight=dist)

# Connect start point to all houses
for i in range(len(houses)):
    dist = np.linalg.norm(start_point - houses[i])
    G.add_edge("Start", f"H{i}", weight=dist)

# ------------------------------------------------------------
# STEP 3: RUN DIJKSTRA FOR ONLY WASTE HOUSES
# ------------------------------------------------------------

# Only keep nodes with waste
waste_nodes = [f"H{i}" for i in range(len(houses)) if waste_sensor[i] == 1]

route = ["Start"]
current = "Start"

total_distance = 0

while waste_nodes:
    # Find nearest waste house
    shortest = {}
    for target in waste_nodes:
        path_length = nx.dijkstra_path_length(G, current, target, weight='weight')
        shortest[target] = path_length

    next_house = min(shortest, key=shortest.get)
    total_distance += shortest[next_house]
    route.append(next_house)
    current = next_house

    waste_nodes.remove(next_house)

# Return to Start
total_distance += nx.dijkstra_path_length(G, current, "Start", weight='weight')
route.append("Start")

# ------------------------------------------------------------
# STEP 4: DISPLAY RESULTS
# ------------------------------------------------------------

print("\nOptimal Waste Collection Route:")
print(" → ".join(route))
print("\nTotal Distance Travelled:", round(total_distance, 2))

# ------------------------------------------------------------
# STEP 5: PLOT
# ------------------------------------------------------------

pos = nx.get_node_attributes(G, 'pos')

plt.figure(figsize=(8, 8))
nx.draw(G, pos, node_size=50, alpha=0.2)

# Highlight the route
path_edges = list(zip(route[:-1], route[1:]))
nx.draw_networkx_nodes(G, pos, nodelist=route, node_color='red', node_size=200)
nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2, edge_color='blue')

plt.title("Waste Collection – KMeans + Dijkstra Route")
plt.show()
