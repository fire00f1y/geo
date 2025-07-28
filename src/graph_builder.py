import time
import json

from torch import nn
import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data


def build_bipartite_graph(csv_path: str, embedding_dim: int = 16) -> Data:
    """Build a bipartite graph from the sample dataset."""

    headers = ['user_id', 'book_id', 'rating', 'timestamp']
    df = pl.read_csv(csv_path, has_header=False)
    df = df.rename({old: new for old, new in zip(df.columns, headers)})

    user_encoder = LabelEncoder()
    book_encoder = LabelEncoder()
    df = df.with_columns([
        pl.Series('user_idx', user_encoder.fit_transform(df['user_id'].to_numpy())),
        pl.Series('book_idx', book_encoder.fit_transform(df['book_id'].to_numpy())),
    ])
    num_users = df['user_idx'].n_unique()
    num_books = df['book_idx'].n_unique()

    # save out the index mappings:
    book_id_list = book_encoder.classes_.tolist()
    user_id_list = user_encoder.classes_.tolist()

    assert set(range(num_books)) == set(df['book_idx'].unique().to_list())
    assert set(range(num_users)) == set(df['user_idx'].unique().to_list())
    assert len(book_id_list) == num_books
    assert len(user_id_list) == num_users

    print('book_idx unique:', df['book_idx'].n_unique())
    print('book_id unique:', df['book_id'].n_unique())
    print('book_idx min/max:', df['book_idx'].min(), df['book_idx'].max())
    print('Mapping file length:', len(book_id_list))
    print('Num_books as in code:', num_books)
    np_bincount = np.bincount(df['book_idx'])
    print('Books with count > 0:', np.sum(np_bincount > 0))

    print(df.head(10))

    with open("data/embeddings/book_id_map.json", "w") as f:
        json.dump(book_id_list, f)
    with open("data/embeddings/user_id_map.json", "w") as f:
        json.dump(user_id_list, f)

    source = torch.tensor(df['user_idx'].to_numpy(), dtype=torch.long)
    target = torch.tensor(df['book_idx'].to_numpy() + num_users, dtype=torch.long)
    edge_index = torch.stack([source, target], dim=0)
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index example: {edge_index[:5]}")

    # normalize and organize edge attributes
    ratings = df['rating'].to_numpy()
    timestamps = df['timestamp'].to_numpy()
    ratings_norm = (ratings - ratings.min()) / (ratings.max() - ratings.min())
    timestamps_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    edge_attr = torch.tensor(np.stack([ratings_norm, timestamps_norm], axis=1), dtype=torch.float)
    print(f"Edge attribute shape: {edge_attr.shape}")
    print(f"Edge attribute example: {edge_attr[:5]}")

    # node_features = torch.zeros(num_users + num_books, 2)
    # node_features[:num_users, 0] = 1
    # node_features[num_users:, 1] = 1
    # print(f"Node features shape: {node_features.shape}")
    # print(f"Node features example: {node_features[:5]}")

    # With node indices that identify each node:
    node_indices = torch.arange(num_users + num_books, dtype=torch.long)
    node_types = torch.zeros(num_users + num_books, dtype=torch.long)
    node_types[num_users:] = 1  # 0 for users, 1 for books

    # Stack them as features (index + type)
    node_features = torch.stack([node_indices, node_types], dim=1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_users + num_books
    )

    data.num_users = num_users
    data.num_books = num_books

    return data


def print_graph_metrics(built_graph: Data):
    num_users = built_graph.num_users
    num_books = built_graph.num_books
    print(f"Num users: {num_users}, Num books: {num_books}")
    degrees = torch.bincount(built_graph.edge_index[1], minlength=num_users + num_books)

    user_degrees = torch.bincount(built_graph.edge_index[0], minlength=num_users)
    print("User degree stats:")
    print(f"  Min:    {user_degrees.min().item()}")
    print(f"  Max:    {user_degrees.max().item()}")
    print(f"  Mean:   {user_degrees.float().mean().item()}")
    print(f"  Median: {user_degrees.float().median().item()}")
    print(f"  Std:    {user_degrees.float().std().item()}")
    print(f"  Count of zero degree: {(user_degrees == 0).sum().item()}")
    print(f"  Total users: {num_users}")

    book_start = num_users
    book_end = num_users + num_books
    book_degrees = degrees[book_start:book_end]
    print("Book degree stats:")
    print(f"  Min: {book_degrees.min()}")
    print(f"  Max: {book_degrees.max()}")
    print(f"  Mean: {book_degrees.float().mean()}")
    print(f"  Std: {book_degrees.float().std()}")
    print(f"  Median: {book_degrees.float().median()}")
    print(f"  Count of zero degree:", (book_degrees == 0).sum().item())
    print(f"  Total items: {book_degrees}")

    print("Graph summary:")
    print(f"  Nodes: {built_graph.num_nodes} (users: {built_graph.num_users}, books: {built_graph.num_books})")
    print(f"  Edges: {built_graph.edge_index.shape[1]}")
    print(f"  Edge attr shape: {built_graph.edge_attr.shape}")
    print(f"  Node features shape: {built_graph.x.shape}")


if __name__ == "__main__":
    start = time.time()
    file_path = "data/amazon_books_ratings.csv"
    graph = build_bipartite_graph(file_path)
    print_graph_metrics(graph)
    torch.save(graph, "data/bipartite_graph.pt")
    print(f"  Time taken: {time.time() - start:.2f}s")
