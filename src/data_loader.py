import time
from typing import Sequence, Tuple

import numpy as np
import polars as pl

from util import download_file


def download_amazon_books():
    url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv"
    output_file_path = "data/amazon_books_ratings.csv"

    return download_file(url, output_file_path, exist_ok=True)


def load_and_process_data(headers: Sequence[str]) -> pl.DataFrame or None:
    print("🚀 Starting Amazon Books dataset loading...")

    # Download dataset
    file_path = download_amazon_books()
    if not file_path:
        return None

    print("\n📊 Loading and analyzing dataset...")
    # Load data with proper column names
    df = pl.read_csv(file_path, has_header=False)
    df = df.rename({old: new for old, new in zip(df.columns, headers)})

    print(f"✓ Raw dataset loaded:")
    print(f"  • Shape: {df.shape}")
    print(f"  • Users: {df['user_id'].n_unique():,}")
    print(f"  • Books: {df['book_id'].n_unique():,}")
    print(f"  • Interactions: {len(df):,}")
    print(f"  • Sparsity: {(1 - len(df) / (df['user_id'].n_unique() * df['book_id'].n_unique())) * 100:.2f}%")

    # Basic data quality check
    print(f"\n🔍 Data quality check:")
    print(f"  • Missing values: {df.null_count().to_series().sum()}")
    print(f"  • Rating range: {df['rating'].min()} - {df['rating'].max()}")
    print(f"  • Rating distribution:\n{df['rating'].value_counts(sort=False).sort('rating')}")

    return df


if __name__ == "__main__":
    start = time.time()
    columns = ['user_id', 'book_id', 'rating', 'timestamp']
    full_data = load_and_process_data(columns)
    print(f"Dataset loading and sample created in {time.time() - start:.2f} seconds.")

    if full_data is not None:
        print("\n🎯 Dataset ready for GNN model development!")
    else:
        print("❌ Dataset loading failed")