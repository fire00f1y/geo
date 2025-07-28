import os
import json

import faiss
import torch
import numpy as np
from src.model import SimpleRecGNN
from util import load_graph, load_saved_graph, load_config
from graph_builder import print_graph_metrics


def generate_embeddings(output_path: str):
    config = load_config()

    gnn_model = SimpleRecGNN(
        input_dim=config["input_dim"],
        num_layers=config["num_layers"],
        hidden_dim=config["hidden_dim"],
        embedding_dim=config["embedding_dim"],
        dropout=config["dropout"]
    )
    gnn_model = gnn_model.cuda()
    gnn_model.load_state_dict(load_saved_graph())
    gnn_model.eval()

    graph = load_graph()
    node_features = graph.x.cuda()
    edge_indices = graph.edge_index.cuda()
    ratings = graph.edge_attr[:, 0].cuda()

    for n, p in gnn_model.named_parameters():
        print(f"n={n}, mean={p.mean().item()}, std={p.std().item()}")

    with torch.no_grad():
        embeddings = gnn_model(node_features, edge_indices, ratings)
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Num users: {graph.num_users}, Num books: {graph.num_books}")

        item_embeddings = embeddings[graph.num_users:]
        print(f"Item embeddings shape: {item_embeddings.shape}")

        item_embeddings_np = item_embeddings.cpu().numpy()
        print(f"Item embeddings numpy shape: {item_embeddings_np.shape}")
        np.save(output_path, item_embeddings_np)
        np.savetxt("data/embeddings/book_embeddings.csv", item_embeddings_np, delimiter=",")

        uniq = np.unique(item_embeddings_np, axis=0)
        print(f"Unique vectors (pre-save): {uniq.shape[0]}, of total: {item_embeddings_np.shape[0]}")


def map_embeddings_to_id(map_file_path: str):
    import polars as pl
    df = pl.read_csv("data/amazon_books_ratings.csv", has_header=False)
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = df.rename({old: new for old, new in zip(df.columns, columns)})

    from sklearn.preprocessing import LabelEncoder
    item_encoder = LabelEncoder()
    item_encoder.fit(df['item_id'].to_numpy())
    unique_item_ids = item_encoder.classes_.tolist()
    print(f"First 5 item_ids: {unique_item_ids[:5]}")
    print(f"Last 5 item_ids: {unique_item_ids[-5:]}")
    print(f"Total unique item_ids: {len(unique_item_ids)}")

    with open(map_file_path, "w") as f:
        json.dump(unique_item_ids, f)


def create_embedding_index(embeddings_path: str,mapping_path: str, index_path: str):
    # Load assets
    embeddings = np.load(embeddings_path)
    with open(mapping_path, "r") as f:
        book_id_map = json.load(f)
    assert embeddings.shape[0] == len(book_id_map)

    dim = embeddings.shape[1]
    print(f"Loaded {embeddings.shape[0]} embeddings of size {dim} (dtype: {embeddings.dtype})")

    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-10, np.inf)

    # Build FAISS Index
    index_flat = faiss.IndexFlatIP(dim)  # inner product == cosine after L2 normalization
    index_flat.add(embeddings)
    print(f"FAISS CPU Index contains {index_flat.ntotal} vectors")

    # Save
    faiss.write_index(index_flat, index_path)
    print(f"FAISS index saved to {index_path}")

    D, I = index_flat.search(embeddings[0:1], 10)
    print(f"Dummy query (indices):    {I[0]}")
    print(f"Dummy query (similarity): {D[0]}")
    print(f"Mapped book IDs:          {[book_id_map[i] for i in I[0]]}")


def explore_embeddings():
    emb = np.load("data/embeddings/book_embeddings.npy")
    print("Embeddings shape:", emb.shape)

    unique_rows = np.unique(emb, axis=0)
    print("Unique vectors:", unique_rows.shape[0], "of total:", emb.shape[0])
    print("Any all-zero embedding?", np.any(np.all(emb == 0, axis=1)))
    print("Mean, std of all vectors:", emb.mean(), emb.std())

    graph = load_graph()
    num_users = graph.num_users
    num_books = graph.num_books
    print(f"Num users: {num_users}, Num books: {num_books}")

    print_graph_metrics(graph)


def create_embeddings_and_index():
    file_path = "data/embeddings/book_embeddings.npy"
    book_mapping_file_path = "data/embeddings/book_id_map.json"
    user_mapping_file_path = "data/embeddings/user_id_map.json"
    index_file_path = "data/embeddings/books_faiss.index"

    if not os.path.exists(file_path):
        generate_embeddings(file_path)
    else:
        print(f"Embeddings file already exists at {file_path}. Skipping generation.")

    if not os.path.exists(book_mapping_file_path):
        map_embeddings_to_id(book_mapping_file_path)
    else:
        print(f"Mapping file already exists at {book_mapping_file_path}. Skipping generation.")

    if not os.path.exists(index_file_path):
        create_embedding_index(file_path, book_mapping_file_path, index_file_path)
    else:
        print(f"Index file already exists at {index_file_path}. Skipping generation.")


if __name__ == "__main__":
    create_embeddings_and_index()
    explore_embeddings()