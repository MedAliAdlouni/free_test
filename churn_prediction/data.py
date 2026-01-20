"""Data loading utilities."""

import yaml
import pandas as pd
from pathlib import Path


def load_data(data_path=None):
    """Load data from YAML or processed CSV."""
    if data_path is None:
        root = Path(__file__).parent.parent
        # Try processed data first
        for ext in ['.csv', '.parquet', '.pkl']:
            path = root / f'processed_data_for_ml{ext}'
            if path.exists():
                if ext == '.csv':
                    return pd.read_csv(path)
                elif ext == '.parquet':
                    return pd.read_parquet(path)
                else:
                    import pickle
                    with open(path, 'rb') as f:
                        return pickle.load(f)
        
        # Fall back to raw YAML
        path = root / 'data.yaml'
        if path.exists():
            with open(path, 'r') as f:
                return pd.DataFrame(yaml.load(f, Loader=yaml.FullLoader))
    
    if data_path.endswith('.yaml') or data_path.endswith('.yml'):
        with open(data_path, 'r') as f:
            return pd.DataFrame(yaml.load(f, Loader=yaml.FullLoader))
    elif data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

