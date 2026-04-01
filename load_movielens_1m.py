import pandas as pd
import numpy as np

def load_movielens_1m(filepath=r"C:\\Users\\ramsi\\Desktop\\matrix_factorization_replication\\ml-1m\\ml-1m\\ratings.dat"):
    """
    Load MovieLens 1M dataset
    
    Format: UserID::MovieID::Rating::Timestamp
    """
    print(f"Loading MovieLens 1M from {filepath}...")
    
    # Read the file (note the :: separator)
    data = pd.read_csv(filepath, 
                       sep='::',
                       engine='python',  # Required for :: separator
                       names=['user_id', 'item_id', 'rating', 'timestamp'],
                       encoding='latin-1')  # Handle special characters
    
    # Convert to zero-indexed (for compatibility with your models)
    data['user_id'] = data['user_id'] - 1
    data['item_id'] = data['item_id'] - 1
    
    print(f"Loaded {len(data):,} ratings")
    print(f"Users: {data['user_id'].nunique()}")
    print(f"Items: {data['item_id'].nunique()}")
    print(f"Ratings range: {data['rating'].min()} to {data['rating'].max()}")
    print(f"Sparsity: {(1 - len(data) / (data['user_id'].nunique() * data['item_id'].nunique())) * 100:.2f}%")
    
    return data

# Test it
if __name__ == "__main__":
    data = load_movielens_1m()
    print("\nFirst few rows:")
    print(data.head())
    print("\nRating distribution:")
    print(data['rating'].value_counts().sort_index())