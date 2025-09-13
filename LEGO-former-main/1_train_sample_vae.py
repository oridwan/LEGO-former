import argparse
import os
import numpy as np
import pandas as pd
import torch
import wandb
from datetime import datetime

from lego.VAE import VAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VAE Table Synthesizer")
    parser.add_argument("--data", default="/users/ksu4/LEGO-former-main/train-v4.csv", help="Input CSV data")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of epochs for training (default: 250)",
    )
    parser.add_argument(
        "--nbatch", 
        type=int, 
        default=256, 
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
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="LEGO-Transformer", 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default= "Transformer_train-v4",
        help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--no_wandb", 
        action="store_true", 
        help="Disable wandb logging"
    )

    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        # Auto-generate run name if not provided
        if args.wandb_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.wandb_name = f"VAE_ep{args.epochs}_bs{args.nbatch}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.nbatch,
                "seed": args.seed,
                "cutoff": args.cutoff,
                "sample_size": args.sample,
                "embedding_dim": 128,
                "compress_dims": (512, 512),
                "decompress_dims": (512, 512),
                "l2scale": 2e-5,
                "loss_factor": 2,
            },
            tags=["VAE", "crystal-generation", "LEGO"]
        )
        print(f"Wandb initialized: {wandb.run.name}")
    
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

    # Log data statistics to wandb
    if not args.no_wandb:
        wandb.config.update({
            "data_shape": df.shape,
            "num_discrete_cols": len(dis_cols),
            "discrete_columns": dis_cols,
            "discrete_cell": discrete_cell,
            "discrete_coords": discrete,
            "num_wps": num_wps,
            "device": "cuda" if cuda else "cpu"
        })

    # Initialize VAE synthesizer
    synthesizer = VAE(
        embedding_dim=256,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        l2scale=2e-5,
        loss_factor=1.0,
        epochs=args.epochs,
        verbose=True,
        cuda=cuda,
        batch_size=args.nbatch,
        folder="models/Transformer_300epochs_classposition",
        wandb_enabled=not args.no_wandb,  # Pass wandb flag to VAE
    )

    # Train the VAE model
    print("Training VAE model...")
    synthesizer.fit(df, discrete_columns=dis_cols)

    # Generate synthetic data
    synthetic_data_size = args.sample if args.sample is not None else len(df)
    print(f"Generating {synthetic_data_size} synthetic samples...")
    df_synthetic = synthesizer.sample(samples=synthetic_data_size)

    print(f"Synthetic data sample:\n{df_synthetic.head(10)}\n")

    # Log synthetic data statistics
    if not args.no_wandb:      
        wandb.log({
            "synthetic_data/sample_size": len(df_synthetic),
            "synthetic_data/num_columns": len(df_synthetic.columns),
        })
        
        # Log sample statistics comparison
        for col in df.columns:
            if col in df_synthetic.columns:
                original_mean = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None
                synthetic_mean = df_synthetic[col].mean() if pd.api.types.is_numeric_dtype(df_synthetic[col]) else None
                
                if original_mean is not None and synthetic_mean is not None:
                    wandb.log({
                        f"comparison/{col}_original_mean": original_mean,
                        f"comparison/{col}_synthetic_mean": synthetic_mean,
                        f"comparison/{col}_mean_diff": abs(original_mean - synthetic_mean)
                    })

    # Save synthetic data
    os.makedirs("data/sample", exist_ok=True)
    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/sample/Transformer-dis{len(dis_cols)}-{synthetic_data_size}-{time}.csv"

    # Clean data for output
    df_synthetic.columns = df_synthetic.columns.str.replace(" ", "")
    df_synthetic = df_synthetic.map(lambda x: str(x).replace(",", " "))
    
    df_synthetic.to_csv(output_file, index=False, header=True)
    print(f"Saved {synthetic_data_size} samples to {output_file}")

    # Log final results
    if not args.no_wandb:
        wandb.log({
            "final/output_file": output_file,
            "final/training_completed": True
        })
        
        # Save the synthetic data as wandb artifact
        artifact = wandb.Artifact(
            name=f"synthetic_data_{wandb.run.name}",
            type="dataset",
            description=f"Synthetic crystal data generated by VAE"
        )
        artifact.add_file(output_file)
        wandb.log_artifact(artifact)
        
        wandb.finish()
        print("Wandb logging completed")