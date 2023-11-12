# script source: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
import itertools

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

path = './data/Cora'
dataset = Planetoid(root=path, name='Cora')
data = dataset[0]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc



@torch.no_grad()
def plot_points(colors,p,q):
    model.eval()
    z = model().cpu().numpy()
    z = TSNE(n_components=2).fit_transform(z)
    y = data.y.cpu().numpy()

    plt.figure(figsize=(20, 15))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i], label=labels[i])
    plt.axis('off')
    plt.legend()
    filename =("node2vec_p"+str(p)+"_q"+str(q)).replace(".", "")
    plt.savefig(filename+".png")
    plt.clf()


colors = [
    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
]
labels = ["Theory", "Reinforcement Learning", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods",
          "Case Based", "Rule Learning"]
# source: https://stellargraph.readthedocs.io/en/v1.0.0rc1/demos/node-classification/gcn/gcn-cora-node-classification-example.html
# then I cross refed data.y
# unique, counts = data.y .unique(return_counts=True)
# count_dict = dict(zip(unique.numpy(), counts.numpy()))

values =[0.2,0.5,1,2,5]
combinations = list(itertools.product(values, repeat=2))
for p, q in combinations:
    model = Node2Vec(
        data.edge_index,
        embedding_dim=128,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=p,
        q=q,
        sparse=True,
    ).to(device)

    num_workers = 0
    loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


    for epoch in range(1, 101):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    plot_points(colors,p,q)

