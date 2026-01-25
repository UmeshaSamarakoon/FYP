import json
import os
import pandas as pd

def load_metadata(data_root):
    """Reads metadata.json from the data/raw folder."""
    metadata_path = os.path.join(data_root, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Could not find metadata at: {metadata_path}")
        
    with open(metadata_path, 'r') as f:
        return json.load(f)

# Example usage:
if __name__ == "__main__":
    df = load_metadata("data/raw/sample_submission.csv")
    print(f"Total videos: {len(df)}")
    print(df['label'].value_counts())
    
    # Show a few "Causal Pairs" (Fake + its Original)
    fakes = df[df['label'] == 'FAKE'].head(3)
    print("\nSample Causal Pairs:")
    print(fakes[['filename', 'original']])