import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGE

from util import load_graph


class SimpleRecGNN(nn.Module):

    def __init__(self, input_dim, hidden_dim=32, embedding_dim=16, num_layers=6, dropout=0.25, activation_function=F.relu):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.convs = nn.ModuleList()
        if num_layers > 1:
            self.convs.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        else:
            self.convs.append(GCNConv(input_dim, embedding_dim))
        self.dropout = dropout
        self.activation = activation_function

    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index, edge_weight)
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index, edge_weight)
        return h


if __name__ == "__main__":
    start_time = time.time()

    ### 1. Load the prebuilt bipartite graph ###
    graph = load_graph()

    ### 2. Move graph data to GPU (for speed) ###
    node_features = graph.x.cuda()  # [num_nodes, num_node_features]
    edge_indices = graph.edge_index.cuda()  # [2, num_edges]

    ### 3. Build the recommendation GNN model ###
    # Initialize with correct feature dimensions
    gnn_model = SimpleRecGNN(input_dim=node_features.shape[1], num_layers=4)
    gnn_model = gnn_model.cuda()  # Ensure model on GPU

    ### 4. Forward pass: calculate node embeddings with no grad ###
    with torch.no_grad():
        all_node_embeddings = gnn_model(node_features, edge_indices)
        # Shape: [num_nodes, embedding_dim]

    ### 5. Retrieve example embeddings for a user and an item ###
    first_user_idx = 0
    first_book_idx = graph.num_users
    user_embedding = all_node_embeddings[first_user_idx].cpu().tolist()
    book_embeddings = all_node_embeddings[first_book_idx].cpu().tolist()

    ### 6. Compute an example predicted score (user prefers item?) ###
    score = torch.dot(
        all_node_embeddings[first_user_idx],
        all_node_embeddings[first_book_idx]
    ).item()

    ### 7. Display results with clear annotations ###
    print(f"User 0 embedding vector (len={len(user_embedding)}): {user_embedding}")
    print(f"Item 0 embedding vector (len={len(book_embeddings)}): {book_embeddings}")
    print("All node embeddings shape:", all_node_embeddings.shape)
    print(f"Predicted user 0 – item 0 similarity score: {score:.4f}")
    print(f"Completed in {time.time() - start_time:.2f} seconds.")
