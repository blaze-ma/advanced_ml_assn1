import torch
from networkx import connected_components
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected
import numpy as np
import random
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx

path = './data/Cora'
dataset = Planetoid(root=path, name='Cora')
data = dataset[0]

# nx_cora = to_networkx(cora, to_undirected=True)
# largest_cc = nx_cora.subgraph(max(nx.connected_components(nx_cora), key=len))
# data = from_networkx(largest_cc)

def majority_voting(node, edge_index, labels):
    neighbors = edge_index[1][edge_index[0] == node]
    neighbor_labels = labels[neighbors]
    unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
    majority_label = unique_labels[counts.argmax()]
    return majority_label

# Initialize list to store accuracies
accuracies = []
known_ratios = np.arange(0.1, 0.31, 0.01)  # Known ratios from 0.1 to 0.3

# Iterate over different known ratios
for known_ratio in known_ratios:
    temp_accuracies = []
    for i in range(10):
        num_nodes = data.num_nodes
        num_known = int(known_ratio * num_nodes)

        # Generate known and unknown masks
        all_nodes = list(range(num_nodes))
        random.shuffle(all_nodes)
        known_nodes = all_nodes[:num_known]
        unknown_nodes = all_nodes[num_known:]

        # Create masks
        known_mask = torch.zeros(num_nodes, dtype=torch.bool)
        known_mask[torch.tensor(known_nodes)] = True
        unknown_mask = ~known_mask

        # Initialize labels for unknown nodes
        inferred_labels = torch.empty(num_nodes, dtype=torch.long)
        inferred_labels[known_mask] = data.y[known_mask]

        # Loopy Belief Propagation (LBP) with Majority Voting
        max_iters = 100
        edge_index = to_undirected(data.edge_index, num_nodes)

        for _ in range(max_iters):
            prev_labels = inferred_labels.clone()
            for node in unknown_nodes:
                inferred_labels[node] = majority_voting(node, edge_index, prev_labels)

            # Check for convergence
            if torch.equal(inferred_labels, prev_labels):
                break

        # Calculate accuracy
        correct = (inferred_labels[unknown_mask] == data.y[unknown_mask]).sum()
        accuracy = correct.item() / unknown_mask.sum().item()
        print("known:"+str(known_ratio)+" accuracy: "+str(accuracy))
        temp_accuracies.append(accuracy)
    accuracies.append(sum(temp_accuracies)/len(temp_accuracies))

# Plotting the accuracy as a line chart
plt.figure(figsize=(10, 6))
plt.plot(known_ratios, accuracies, marker='o')
plt.title('Accuracy vs Known Ratio')
plt.xlabel('Known Ratio')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig("uncut_lpb.png")
plt.show()
