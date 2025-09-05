import argparse
import os
import numpy as np
import pandas as pd
import torch

from lego.VAE import VAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Table Synthesizer")
    parser.add_argument("--data", required=True, help="Input CSV data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Number of epochs for training (default: 250)",
    )
    parser.add_argument(
        "--nbatch", 
        type=int, 
        default=500, 
        help="Number of batch size for training"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Training seeds"
    )
    parser.add_argument(
        "--cutoff", 
        type=int, 
        help="Cutoff number for training samples"
    )
    parser.add_argument(
        "--sample", 
        type=int, 
        default=100000, 
        help="Output sample size"
    )

    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check CUDA availability
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(args.seed)
        print("CUDA is available")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    
    # Read and prepare data
    df = pd.read_csv(args.data)
    if args.cutoff is not None and len(df) > args.cutoff:
        print(f"Using only {args.cutoff} samples for quick test")
        df = df[:args.cutoff]

    print(f"Data shape: {df.shape}")
    print(f"Data head:\n{df.head()}\n")

    # Set up discrete columns
    dis_cols = ["spg"]
    num_wps = int((len(df.columns) - 7) / 4)
    
    # Check if cell parameters are discrete
    if abs(df['a'][0] - round(df['a'][0])) < 1e-2: 
        discrete_cell = True
        dis_cols.extend(['a', 'b', 'c', 'alpha', 'beta', 'gamma'])
        print("Cell parameters treated as discrete")
    else:
        discrete_cell = False
        print("Cell parameters treated as continuous")

    # Check if coordinates are discrete
    discrete = df['x0'].max() >= 2.5 + 1e-3

    for i in range(num_wps): 
        dis_cols.append('wp' + str(i))
        if discrete:
            dis_cols.append('x' + str(i))
            dis_cols.append('y' + str(i))
            dis_cols.append('z' + str(i))

    print(f"Discrete columns: {dis_cols}")

    # Initialize VAE synthesizer
    synthesizer = VAE(
        embedding_dim=128,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        l2scale=1e-5,
        loss_factor=2,
        epochs=args.epochs,
        verbose=True,
        cuda=cuda,
        batch_size=args.nbatch,
        folder="models/VAE",
    )

    # Train the VAE model
    print("Training VAE model...")
    synthesizer.fit(df, discrete_columns=dis_cols)

    # Generate synthetic data
    synthetic_data_size = args.sample if args.sample is not None else len(df)
    print(f"Generating {synthetic_data_size} synthetic samples...")
    df_synthetic = synthesizer.sample(samples=synthetic_data_size)

    print(f"Synthetic data sample:\n{df_synthetic.head(10)}\n")

    # Save synthetic data
    os.makedirs("data/sample", exist_ok=True)
    output_file = f"data/sample/VAE-dis{len(dis_cols)}-{synthetic_data_size}.csv"
    
    # Clean data for output
    df_synthetic.columns = df_synthetic.columns.str.replace(" ", "")
    df_synthetic = df_synthetic.map(lambda x: str(x).replace(",", " "))
    
    df_synthetic.to_csv(output_file, index=False, header=True)
    print(f"Saved {synthetic_data_size} samples to {output_file}")