import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from model import SimpleRecGNN
from util import load_graph


def split_data(data: Data, split_ratio: float = 0.1) -> tuple:
    """Split the bipartite graph into train and test sets."""
    num_edges = data.edge_index.shape[1]
    all_edge_indices = torch.arange(num_edges)
    all_edge_indices = all_edge_indices.cpu().numpy()
    np.random.shuffle(all_edge_indices)

    split_train = int(num_edges * (1.0 - (split_ratio * 2)))
    split_val = int(num_edges * (1.0 - split_ratio))

    return all_edge_indices[:split_train], all_edge_indices[split_train:split_val], all_edge_indices[split_val:]


def train_model(layer_count: int, hidden_dim_count: int, dropout_rate: float, learning_rate: float, optimizer_decay: float, epochs: int = 400, embedding_dim_count: int = 16, verbose: bool = False) -> tuple[float, float]:
    start = time.time()
    graph = load_graph()
    train_edges, val_edges, test_edges = split_data(graph)
    ratings = graph.edge_attr[:, 0]
    ratings_gpu = ratings.cuda()

    node_features = graph.x.cuda()
    edge_indices = graph.edge_index.cuda()
    gnn_model = SimpleRecGNN(input_dim=node_features.shape[1], num_layers=layer_count,
                             hidden_dim=hidden_dim_count, embedding_dim=embedding_dim_count,
                             dropout=dropout_rate, num_users=graph.num_users, num_books=graph.num_books)
    gnn_model = gnn_model.cuda()

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate, weight_decay=optimizer_decay)

    final_train_loss = float('inf')
    final_val_loss = float('inf')
    best_epoch_val_loss = float('inf')
    best_epoch_train_loss = float('inf')
    best_epoch_train = 0
    best_epoch_validation = 0

    early_stop = False
    last_epoch = 0
    patience = 20
    min_delta = 1e-4
    no_improvement_count = 0

    for epoch in range(epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # forward pass
        node_embeddings = gnn_model(node_features, edge_indices, ratings_gpu)

        # compute scores
        src_idx = graph.edge_index[0, train_edges]
        target_idx = graph.edge_index[1, train_edges]
        score_prediction = (node_embeddings[src_idx] * node_embeddings[target_idx]).sum(dim=1)
        score_actual = ratings[train_edges].to(node_embeddings.device)

        # calculate loss
        loss = F.mse_loss(score_prediction, score_actual)

        # backprop
        loss.backward()
        optimizer.step()

        # report loss for debugging
        gnn_model.eval()
        with torch.no_grad():
            val_src = graph.edge_index[0, val_edges]
            val_tgt = graph.edge_index[1, val_edges]
            val_pred = (node_embeddings[val_src] * node_embeddings[val_tgt]).sum(dim=1)
            val_true = ratings[val_edges].to(node_embeddings.device)
            val_loss = F.mse_loss(val_pred, val_true)

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

        gnn_model.train()

        last_epoch = epoch + 1
        final_val_loss = val_loss.item()
        final_train_loss = loss.item()

        # early stopping logic
        if val_loss.item() < best_epoch_val_loss - min_delta:
            best_epoch_val_loss = val_loss.item()
            best_epoch_train_loss = loss.item()
            best_epoch_validation = last_epoch
            best_epoch_train = last_epoch
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            if verbose:
                print(f"No improvement for {patience} epochs. Stopping training at epoch {last_epoch}.")
            early_stop = True
            break

    duration = time.time() - start
    if verbose:
        print(f"Completed in {duration:.2f} seconds.")

    torch.save(gnn_model.state_dict(), "models/book_recs.pth")

    import json

    training_metadata = {
        "validation": {
            "train_loss": final_train_loss,
            "validation_loss": final_val_loss,
            "training_size": len(train_edges),
            "validation_size": len(val_edges),
            "test_size": len(test_edges),
        },
        "optimizer": {
            "optimizer": "Adam",
            "lr": learning_rate,
            "weight_decay": optimizer_decay,
        },
        "loss": "MSE",
        "epochs": {
            "last_epoch": last_epoch,
            "epochs": epochs,
            "best_epoch_validation": best_epoch_validation,
            "best_epoch_train": best_epoch_train,
            "best_epoch_val_loss": best_epoch_val_loss,
            "best_epoch_train_loss": best_epoch_train_loss,
            "early_stop": early_stop,
        },
        "training_time": f"{duration:.2f}s",
    }
    gnn_model.config["training_metadata"] = training_metadata
    with open(f"models/model_config.json", "w") as f:
        json.dump(gnn_model.config, f)
        f.write('\n')

    for n, p in gnn_model.named_parameters():
        print(f"n={n}, mean={p.mean().item()}, std={p.std().item()}")

    return final_train_loss, final_val_loss


def grid_search():
    start_total = time.time()

    count = 0
    hidden_dim_counts = [32]
    layer_counts = [6]
    dropout_rates = [0.3]
    learning_rates = [0.001, 0.0015]
    optimizer_decays = [1e-4]
    per_config_runs = 3
    verbose = False

    for decay in optimizer_decays:
        for lr in learning_rates:
            for layer in layer_counts:
                for dr in dropout_rates:
                    for hidden_dim in hidden_dim_counts:
                        for index in range(per_config_runs):
                            run_start = time.time()
                            if verbose:
                                print(f"Training model {count + 1} {index + 1}/{per_config_runs}...")
                                print(f"Model config: hidden_dim_count={hidden_dim}, layer_count={layer}, dropout_rate={dr}, learning_rate={lr}, optimizer_decay={decay}...")
                            train_final, val_final = train_model(layer_count=layer, hidden_dim_count=hidden_dim,
                                                                 dropout_rate=dr, learning_rate=lr,
                                                                 optimizer_decay=decay, verbose=verbose)
                            print(f"Model {count + 1}:{index + 1} completed in {time.time() - run_start:.2f}s. Final train loss: {train_final:.4f}, final val loss: {val_final:.4f}...")
                            count += 1
    print(f"Total training time: {time.time() - start_total:.2f}s for {count} permutations")


if __name__ == "__main__":
    # grid_search()
    train_model(layer_count=2, hidden_dim_count=32, embedding_dim_count=16, dropout_rate=0.25,
                learning_rate=0.001, optimizer_decay=1e-5, epochs=400, verbose=True)
