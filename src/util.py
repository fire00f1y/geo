import os
import json

import requests
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def download_file(url: str, file_path: str, exist_ok: bool = True) -> str or None:
    """Download Amazon Books ratings dataset."""

    # Create data directory
    os.makedirs('data', exist_ok=exist_ok)

    if os.path.exists(file_path):
        print(f"✓ File already exists at {file_path}")
        return file_path

    print("📥 Downloading Amazon Books dataset...")
    print(f"Source: {url}")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(file_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✅ Dataset downloaded successfully to {file_path}")
        return file_path

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


def load_graph() -> Data:
    return torch.load('data/bipartite_graph.pt', weights_only=False)


def load_saved_graph():
    return torch.load("models/book_recs.pth")


def load_config():
    with open("models/model_config.json") as f:
        config = json.load(f)
        return config