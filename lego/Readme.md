# DataTransformer Quick Guide

Transform crystal structure data using Bayesian GMM for continuous columns and One-Hot encoding for discrete columns.

## Quick Setup

```python
import pandas as pd
from Data_transformer_fix_gmm import DataTransformer

# Load data
df = pd.read_csv("train-v4.csv").iloc[:, :-2]

# Define columns
discrete_columns = ["spg"] + [f"wp{i}" for i in range(8)]
xyz_columns = [f"{axis}{idx}" for idx in range(8) for axis in ("x", "y", "z")]

# Group columns (optional - improves performance)
grouped_discrete_columns = [discrete_columns[1:]]  # Group wp0-wp7
grouped_continuous_columns = [xyz_columns]         # Group x,y,z coordinates

# Initialize transformer
transformer = DataTransformer(
    max_clusters=10,
    weight_threshold=0.00001,
    grouped_continuous_columns=grouped_continuous_columns,
    grouped_discrete_columns=grouped_discrete_columns,
)

# Fit and transform
transformer.fit(df, discrete_columns=discrete_columns)
transformed = transformer.transform(df)
recovered = transformer.inverse_transform(transformed)
```

## Key Parameters

- **`max_clusters`**: Max GMM components (default: 10)
- **`weight_threshold`**: Min component weight (default: 0.00001) 
- **`grouped_*_columns`**: Group related columns to share transformers

## Advanced Usage

```python
# Custom grouping for different column types
grouped_continuous_columns = [
    ["a", "b", "c"],           # Unit cell dimensions
    ["alpha", "beta", "gamma"], # Unit cell angles  
    xyz_columns,               # Atomic coordinates
]

# Add noise for generation tasks
sigmas = np.random.normal(0, 0.1, size=transformed.shape[1])
recovered_with_noise = transformer.inverse_transform(transformed, sigmas=sigmas)
```

## Validation

```bash
# Check transformation quality
python check_data_trasnformer_fix_gmm.py --data-path your_data.csv --sample 1000
```
