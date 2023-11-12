from torch_geometric.datasets import Planetoid
from itertools import combinations
from multiprocessing import Pool
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from collections import Counter
import pandas as pd

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import colorsys
from sklearn.cluster import KMeans
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn import metrics


def generate_hex_colors(n):
    if n <= 0:
        raise ValueError("Number of colors must be a positive integer")

    colors = []
    for i in range(n):
        # Divide the color wheel into equal parts
        hue = i / n
        # Convert HSL to RGB
        rgb = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        # Convert RGB to Hex
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)

    return colors
def calculate_lcc_for_node(args):
    node, edge_index = args
    neighbors = []
    for index, value in enumerate(edge_index[0]):
        if value == node:
            neighbors.append(edge_index[1][index])
    node_degree = len(neighbors)
    pairs = combinations(neighbors, 2)
    neighbor_link_count = 0
    for pair in pairs:
        if (pair[0], pair[1]) in zip(edge_index[0], edge_index[1]):
            neighbor_link_count += 1
    if node_degree > 1:
        return (2 * neighbor_link_count) / (node_degree * (node_degree - 1))
    else:
        return 0

#source: https://stackoverflow.com/questions/46258657/spectral-clustering-a-graph-in-python
def spectral_clustering(G, n_clusters):
    nodes_list = list(G.nodes())
    adj_matrix = nx.adjacency_matrix(G)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(adj_matrix)

    clusters = [[] for _ in range(n_clusters)]
    for node, label in zip(nodes_list, sc.labels_):
        clusters[label].append(node)

    plt.figure(figsize=(15, 10))
    colors = generate_hex_colors(n_clusters)
    pos = nx.spring_layout(G)

    nx.draw(G, pos, edge_color='k', with_labels=False, node_size=1, width=0.1)
    for cluster, color in zip(clusters, colors):
         nx.draw_networkx_nodes(G, pos, nodelist=cluster, node_color=color, node_size= 1)
    plt.savefig("spectral_"+str(n_clusters)+".png")
    plt.show()
    return clusters


def generate_spectral_latex_table(cluster_count, clusters, node_label_ids):
    # Labels for mapping
    labels = ["Theory", "Reinforcement Learning", "Genetic Algorithms", "Neural Networks",
              "Probabilistic Methods", "Case Based", "Rule Learning"]
    # source: https://stellargraph.readthedocs.io/en/v1.0.0rc1/demos/node-classification/gcn/gcn-cora-node-classification-example.html
    # then I cross refed data.y
    # unique, counts = data.y .unique(return_counts=True)
    # count_dict = dict(zip(unique.numpy(), counts.numpy()))
    # Select top 10 largest communities
    top_communities = sorted(clusters, key=len, reverse=True)[:10]

    # Initialize data for the LaTeX table
    table_data = []

    for idx, community in enumerate(top_communities, 1):
        # Map community members to node_label_ids
        mapped_labels = [labels[node_label_ids[i]] for i in community]

        # Count the occurrences of each label
        label_counts = Counter(mapped_labels)

        # Find the most common label and its ratio
        most_common_label, count = label_counts.most_common(1)[0]
        label_ratio = (count / len(community)) * 100

        # Add data to the table
        table_data.append({
            "Cluster ID": idx,
            "Cluster Size": len(community),
            "Most Common Label": most_common_label,
            "Label Ratio (%)": round(label_ratio, 2)
        })

    # Create a DataFrame for easier LaTeX conversion
    df = pd.DataFrame(table_data)
    caption = f"Spectral Clustering, cluster count = {cluster_count}"
    return df.to_latex(index=False, caption=caption)

def generate_louvian_latex_table(resolution, communities, node_label_ids):
    # Labels for mapping
    labels = ["Theory", "Reinforcement Learning", "Genetic Algorithms", "Neural Networks",
              "Probabilistic Methods", "Case Based", "Rule Learning"]
    # source: https://stellargraph.readthedocs.io/en/v1.0.0rc1/demos/node-classification/gcn/gcn-cora-node-classification-example.html
    # then I cross refed data.y
    # unique, counts = data.y .unique(return_counts=True)
    # count_dict = dict(zip(unique.numpy(), counts.numpy()))
    # Select top 10 largest communities
    top_communities = sorted(communities, key=len, reverse=True)[:10]

    # Initialize data for the LaTeX table
    table_data = []

    for idx, community in enumerate(top_communities, 1):
        # Map community members to node_label_ids
        mapped_labels = [labels[node_label_ids[i]] for i in community]

        # Count the occurrences of each label
        label_counts = Counter(mapped_labels)

        # Find the most common label and its ratio
        most_common_label, count = label_counts.most_common(1)[0]
        label_ratio = (count / len(community)) * 100

        # Add data to the table
        table_data.append({
            "Community ID": idx,
            "Community Size": len(community),
            "Most Common Label": most_common_label,
            "Label Ratio (%)": round(label_ratio, 2)
        })

    # Create a DataFrame for easier LaTeX conversion
    df = pd.DataFrame(table_data)
    caption = f"Louvian method, resolution = {resolution}"
    return df.to_latex(index=False, caption=caption)

def calculate_average_clustering_coefficient(graph):
    num_of_nodes = graph.num_nodes
    edge_index = graph.edge_index
    pool = Pool()
    args = [(node, edge_index) for node in range(num_of_nodes)]
    LCCs = pool.map(calculate_lcc_for_node, args)
    pool.close()
    pool.join()
    sum_of_lccs = sum(LCCs)
    return sum_of_lccs / num_of_nodes


def plot_degree_distribution(edge_index, number_of_nodes):
    degrees = np.zeros((number_of_nodes,), dtype=int)
    for index, value in enumerate(edge_index[0]):
        degrees[value] += 1
    plt.figure(figsize=(10, 5))
    plt.hist(degrees, bins=range(1, degrees.max() + 1), color='blue', alpha=0.7)
    plt.title('Degree Distribution of Cora Dataset')
    plt.xlabel('Degree')
    plt.ylabel('Count')
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    path = './data/Cora'

    # Loading the Cora dataset
    dataset = Planetoid(root=path, name='Cora')

    # The dataset is a list with a single Data object
    cora = dataset[0]

    # Section 1
    num_of_nodes = cora.num_nodes
    num_of_edges = cora.num_edges / 2
    density = (2 * num_of_edges) / (num_of_nodes * (num_of_nodes - 1))

    print(f'Number of nodes: {num_of_nodes}')
    print(f'Number of edges: {num_of_edges}')
    print(f'Density: {density}')

    avg_degree = 2 * num_of_edges / num_of_nodes
    print(f'Average degree: {avg_degree}')

    # Takes forever to run and uses the whole CPU, only uncomment it before a coffee break.
    # avg_clustering_coeff = calculate_average_clustering_coefficient(cora)
    # print(f'Average clustering coefficient: {avg_clustering_coeff}')
    # plot_degree_distribution(cora.edge_index, num_of_nodes)


# Section 2
    nx_cora = to_networkx(cora, to_undirected=True)
    betweenness_centralities  = nx.betweenness_centrality(nx_cora)

    avg_btwn_centr = sum(betweenness_centralities.values()) / num_of_nodes
    max_btwn_centr = max(betweenness_centralities.values())

    print(f'Average betweenness centrality: {avg_btwn_centr}')
    print(f'Highest betweenness centrality: {max_btwn_centr}')

    eigenvector_centralities  = nx.eigenvector_centrality(nx_cora)
    avg_eigen_centr = sum(eigenvector_centralities.values()) / num_of_nodes
    max_eigen_centr = max(eigenvector_centralities.values())

    print(f'Average eigenvector centrality: {avg_eigen_centr}')
    print(f'Highest eigenvector centrality: {max_eigen_centr}')

# Section 4
    lv_resolution = 2
    lv_communities = nx.community.louvain_communities(nx_cora, resolution=lv_resolution)
    colors = generate_hex_colors(len(lv_communities))
    pos = nx.spring_layout(nx_cora)
    plt.figure(figsize=(15, 10))

    nx.draw(nx_cora, pos, edge_color='k',  with_labels=False, node_size= 1, width= 0.1)

    for community, color in zip(lv_communities, colors):
        nx.draw_networkx_nodes(nx_cora, pos, nodelist=community, node_color=color, node_size= 1)

    table_txt = generate_louvian_latex_table(lv_resolution, lv_communities, cora.y)

    print(len(lv_communities))
    plt.savefig('louvian1000.png')
    plt.show()

    plt.clf()
    clusters = spectral_clustering(nx_cora, 7)
    table = generate_spectral_latex_table(7, clusters, cora.y)
    print("hi")

