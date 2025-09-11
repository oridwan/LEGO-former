"""
VAE module.
"""
import os
from turtle import st
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


#random.seed(42)
#np.random.seed(42)
#torch.manual_seed(42)

class Encoder(Module):
    """Encoder for the VAE.

    Args:
        data_dim (int): Dimensions of the data.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        embedding_dim (int): Size of the output vector.
        transformer (DataTransformer): Data transformer containing column information.
    """

    def __init__(self, data_dim, compress_dims, embedding_dim, transformer):
        super(Encoder, self).__init__()
        self.transformer = transformer
        self.embedding_dim = embedding_dim
        
        # 分离存储：模块列表 + 类型信息列表
        self.column_mlps = torch.nn.ModuleList()  
        self.column_types = []  # 单独存储类型信息
        
        # Calculate total number of tokens
        self.total_tokens = 0
        self.token_info = []
        
        for column_info in transformer._column_transform_info_list:
            if column_info.column_type == 'continuous':
                # Continuous columns are split into two tokens
                total_dim = column_info.output_dimensions
                continuous_dim = 1  # Normalized continuous value
                discrete_dim = total_dim - 1  # One-hot encoded cluster part
                
                # MLP for continuous values (first token)
                continuous_mlp = Sequential(
                    Linear(continuous_dim, embedding_dim // 2),
                    ReLU(),
                    Linear(embedding_dim // 2, embedding_dim)
                )
                
                # MLP for discrete values (second token)
                discrete_mlp = Sequential(
                    Linear(discrete_dim, embedding_dim // 2),
                    ReLU(), 
                    Linear(embedding_dim // 2, embedding_dim)
                )
                
                # 只存储模块，类型信息单独存储
                self.column_mlps.append(continuous_mlp)
                self.column_mlps.append(discrete_mlp)
                
                # 存储类型信息
                self.column_types.append('continuous_value')
                self.column_types.append('continuous_cluster')
                
                # Store token information
                self.token_info.append({
                    'type': 'continuous_value',
                    'start_dim': 0,
                    'dim': continuous_dim
                })
                self.token_info.append({
                    'type': 'continuous_cluster', 
                    'start_dim': continuous_dim,
                    'dim': discrete_dim
                })
                
                self.total_tokens += 2
                
            else:
                # Purely discrete values (single token)
                column_dim = column_info.output_dimensions
                mlp = Sequential(
                    Linear(column_dim, embedding_dim // 2),
                    ReLU(),
                    Linear(embedding_dim // 2, embedding_dim)
                )
                
                # 只存储模块
                self.column_mlps.append(mlp)
                self.column_types.append('discrete')
                
                self.token_info.append({
                    'type': 'discrete',
                    'start_dim': 0,
                    'dim': column_dim
                })
                
                self.total_tokens += 1   
        # Positional encodings
        self.positional_encodings = Parameter(torch.randn(self.total_tokens, embedding_dim))
        
        # Self-attention layers (4 layers)
        self.attention_layers = torch.nn.ModuleList()
        for _ in range(4):
            attention_layer = torch.nn.ModuleDict({
                'attention': torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                ),
                'norm1': torch.nn.LayerNorm(embedding_dim),
                'mlp': Sequential(
                    Linear(embedding_dim, embedding_dim * 4),
                    ReLU(),
                    torch.nn.Dropout(0.1),
                    Linear(embedding_dim * 4, embedding_dim),
                    torch.nn.Dropout(0.1)
                ),
                'norm2': torch.nn.LayerNorm(embedding_dim)
            })
            self.attention_layers.append(attention_layer)
        
        # MLP for each token to generate mu and logvar
        self.token_mu_mlps = torch.nn.ModuleList()
        self.token_logvar_mlps = torch.nn.ModuleList()
        
        for _ in range(self.total_tokens):
            # MLP for mu
            mu_mlp = Sequential(
                Linear(embedding_dim, embedding_dim // 2),
                ReLU(),
                Linear(embedding_dim // 2, embedding_dim)
            )
            self.token_mu_mlps.append(mu_mlp)
            
            # MLP for logvar
            logvar_mlp = Sequential(
                Linear(embedding_dim, embedding_dim // 2),
                ReLU(),
                Linear(embedding_dim // 2, embedding_dim)
            )
            self.token_logvar_mlps.append(logvar_mlp)

    def forward(self, input_):
        """Encode the passed `input_`."""
        tokens = []
        st = 0
        token_idx = 0
        
        for i, column_transform_info in enumerate(self.transformer._column_transform_info_list):
            dim = column_transform_info.output_dimensions
            column_data = input_[:, st : st + dim]
            
            if column_transform_info.column_type == 'continuous':
                # Split continuous column into two tokens
                continuous_part = column_data[:, :1]  # Normalized value
                discrete_part = column_data[:, 1:]    # One-hot cluster ID
                
                # Process continuous value token
                continuous_token = self.column_mlps[token_idx](continuous_part)
                continuous_token = continuous_token + self.positional_encodings[token_idx]
                tokens.append(continuous_token)
                token_idx += 1
                
                # Process cluster ID token
                discrete_token = self.column_mlps[token_idx](discrete_part)
                discrete_token = discrete_token + self.positional_encodings[token_idx]
                tokens.append(discrete_token)
                token_idx += 1
                
            else:
                # Process discrete column as single token
                token = self.column_mlps[token_idx](column_data)
                token = token + self.positional_encodings[token_idx]
                tokens.append(token)
                token_idx += 1
            
            st += dim
        
        
        # Stack tokens to get [batch_size, total_tokens, embedding_dim]
        x = torch.stack(tokens, dim=1)
        
        # Apply self-attention layers
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # MLP with residual connection
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
        
        # Generate mu and logvar for each token
        token_mus = []
        token_logvars = []
        
        for i in range(x.shape[1]):  # for each token
            token_feature = x[:, i, :]  # [batch_size, embedding_dim]
            mu = self.token_mu_mlps[i](token_feature)  # [batch_size, embedding_dim]
            logvar = self.token_logvar_mlps[i](token_feature)  # [batch_size, embedding_dim]
            token_mus.append(mu)
            token_logvars.append(logvar)
        
        # Stack to get [batch_size, total_tokens, embedding_dim]
        mu = torch.stack(token_mus, dim=1)
        logvar = torch.stack(token_logvars, dim=1)
        std = torch.exp(0.5 * logvar)
        
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int): Size of the input vector.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        data_dim (int): Dimensions of the data.
        transformer (DataTransformer): Data transformer containing column information.
        positional_encodings (Parameter): Positional encodings from encoder.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim, transformer=None, positional_encodings=None):
        super(Decoder, self).__init__()
        self.transformer = transformer
        self.positional_encodings = positional_encodings
        self.embedding_dim = embedding_dim
        
        # Self-attention layers (4 layers, same as encoder)
        self.attention_layers = torch.nn.ModuleList()
        for _ in range(4):
            attention_layer = torch.nn.ModuleDict({
                'attention': torch.nn.MultiheadAttention(
                    embed_dim=embedding_dim,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True
                ),
                'norm1': torch.nn.LayerNorm(embedding_dim),
                'mlp': Sequential(
                    Linear(embedding_dim, embedding_dim * 4),
                    ReLU(),
                    torch.nn.Dropout(0.1),
                    Linear(embedding_dim * 4, embedding_dim),
                    torch.nn.Dropout(0.1)
                ),
                'norm2': torch.nn.LayerNorm(embedding_dim)
            })
            self.attention_layers.append(attention_layer)
        
        # 修复：分离模块存储和类型信息
        self.token_decode_mlps = torch.nn.ModuleList()
        self.token_types = []  # 单独存储类型信息
        self.token_info = []
        self.total_tokens = 0
        
        if transformer is not None:
            for column_info in transformer._column_transform_info_list:
                if column_info.column_type == 'continuous':
                    # Continuous columns are split into two tokens
                    total_dim = column_info.output_dimensions
                    continuous_dim = 1
                    discrete_dim = total_dim - 1
                    
                    # MLP for continuous value token
                    continuous_mlp = Sequential(
                        Linear(embedding_dim, embedding_dim // 2),
                        ReLU(),
                        Linear(embedding_dim // 2, continuous_dim)
                    )
                    
                    # MLP for discrete cluster token
                    discrete_mlp = Sequential(
                        Linear(embedding_dim, embedding_dim // 2),
                        ReLU(),
                        Linear(embedding_dim // 2, discrete_dim)
                    )
                    
                    self.token_decode_mlps.append(continuous_mlp)
                    self.token_decode_mlps.append(discrete_mlp)
                    
                    # 分离存储类型信息
                    self.token_types.append('continuous_value')
                    self.token_types.append('continuous_cluster')
                    
                    # Store token information
                    self.token_info.append({
                        'type': 'continuous_value',
                        'dim': continuous_dim,
                        'column_idx': len([t for t in self.token_info if 'column_idx' in t])
                    })
                    self.token_info.append({
                        'type': 'continuous_cluster',
                        'dim': discrete_dim,
                        'column_idx': len([t for t in self.token_info if 'column_idx' in t])
                    })
                    
                    self.total_tokens += 2
                    
                else:
                    # Purely discrete values
                    column_dim = column_info.output_dimensions
                    mlp = Sequential(
                        Linear(embedding_dim, embedding_dim // 2),
                        ReLU(),
                        Linear(embedding_dim // 2, column_dim)
                    )
                    
                    self.token_decode_mlps.append(mlp)
                    self.token_types.append('discrete')
                    
                    self.token_info.append({
                        'type': 'discrete',
                        'dim': column_dim,
                        'column_idx': len([t for t in self.token_info if 'column_idx' in t])
                    })
                    
                    self.total_tokens += 1
        
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_` through self-attention layers."""
        # input_ shape: [batch_size, num_columns, embedding_dim]
        # positional_encodings shape: [num_columns, embedding_dim]
        input_ = input_ + self.positional_encodings.unsqueeze(0)  # Broadcast over batch dimension
        
        # Apply self-attention layers
        x = input_
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # MLP with residual connection
            mlp_out = layer['mlp'](x)
            x = layer['norm2'](x + mlp_out)
        
        # Decode each token back to its original dimensions
        
        reconstructed_columns = []
        token_idx = 0
        
        for column_info in self.transformer._column_transform_info_list:
            if column_info.column_type == 'continuous':
                # Continuous columns have two tokens: value and cluster
                continuous_token = x[:, token_idx, :]  # [batch_size, embedding_dim]
                cluster_token = x[:, token_idx + 1, :]  # [batch_size, embedding_dim]
                
                # Decode both tokens
                continuous_part = self.token_decode_mlps[token_idx](continuous_token)  # [batch_size, 1]
                cluster_part = self.token_decode_mlps[token_idx + 1](cluster_token)  # [batch_size, discrete_dim]
                
                # Concatenate continuous value and cluster parts
                column_output = torch.cat([continuous_part, cluster_part], dim=1)
                reconstructed_columns.append(column_output)
                
                token_idx += 2
            
            else:
                # Discrete columns have single token
                token = x[:, token_idx, :]  # [batch_size, embedding_dim]
                column_output = self.token_decode_mlps[token_idx](token)  # [batch_size, column_dim]
                reconstructed_columns.append(column_output)
                
                token_idx += 1
        
        # Concatenate all column outputs to get final data
        output = torch.cat(reconstructed_columns, dim=1)  # [batch_size, data_dim] still become data_dim
            
        return output, self.sigma

def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    
    # mu, logvar shape: [batch_size, num_columns, embedding_dim]
    # Need to sum over all dimensions except batch dimension, then average over batch
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=(1, 2))  # Sum over columns and embedding dims
    KLD = torch.mean(KLD)  # Average over batch dimension

    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]


class VAE(BaseSynthesizer):
    """
    Variational Autoencoder (VAE) for tabular data.

    Args:
        embedding_dim (int): Size of the output vector.
        compress_dims (tuple or list of ints): Size of each hidden layer.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        l2scale (float): L2 regularization factor.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs for training.
        loss_factor (float): Loss factor for the VAE.
        cuda (bool or str): Device to use ('cuda' or 'cpu').
        verbose (bool): Verbosity flag.
        folder (str): Folder to save models and samples.
    """
    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        loss_factor=2,
        epochs=300,
        lr=1e-3,
        verbose=True,
        cuda=True,
        batch_size=500,
        folder="models/VAE",
        wandb_enabled=False,
    ):
        # 在初始化开始就设置正确的 type
        self.type = "VAE"
        
        # 其他初始化代码...
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.factor = loss_factor
        self.epochs = epochs
        self.lr = 2e-5              # 降低学习率
        self.verbose = verbose
        self.batch_size = batch_size
        self.folder = folder
        self.wandb_enabled = wandb_enabled

        
        
        self.lr_scheduler_step = 100      # 学习率调度步长
        self.lr_scheduler_gamma = 0.9     # 学习率衰减因子

        # β调度参数
        self.beta_start = 0.0
        self.beta_end = 1.0
        self.beta_warmup_epochs = 50  # 前50个epoch逐渐增加β

        # 设置设备
        if cuda and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        # 现在这行应该能正常工作
        # self.type_folder = os.path.join(self.root_folder, self.type)
        self.root_folder = folder
        self.type_folder = os.path.join(self.root_folder, self.type)
        self.samples_folder = os.path.join(self.type_folder, 'samples')
        self.model_folder = os.path.join(self.type_folder, 'models')
        os.makedirs(self.root_folder, exist_ok=True)
        os.makedirs(self.type_folder, exist_ok=True)
        os.makedirs(self.samples_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

    def plot_losses(self, loss_values, filename='loss_plot.png'):
        # Plot the loss values over iterations
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values.index, loss_values['Loss'], label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.root_folder, filename))
        plt.close()

    def save(self, filepath):
        """
        Save the trained model to a file.
        """
        with open(filepath, 'wb') as f:
            joblib.dump({
                'encoder': self.encoder,
                'decoder': self.decoder,
                'transformer': self.transformer,
                'device': self._device
            }, f)

    def load(self, filepath):
        """
        Load a trained model from a file.
        """
        with open(filepath, 'rb') as f:
            state = joblib.load(f)
            self.encoder = state['encoder']
            self.decoder = state['decoder']
            self.transformer = state['transformer']
            self._device = state['device']
            self.encoder.to(self._device)
            self.decoder.to(self._device)
        return self
    
    def get_beta(self, epoch):
        """
        Get the current beta value for the VAE loss.
        """
        if epoch < self.beta_warmup_epochs:
            return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.beta_warmup_epochs)
        return self.beta_end


    @random_state
    def fit(self, train_data, discrete_columns=()):
        """
        Fit the TVAE Synthesizer models to the training data.

        Args:
            train_data (np.ndarray or pd.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        print("Train Data", np.shape(train_data), '\n', train_data[:10])    # [63114, 525]
        print("Transformed Data shape", np.shape(train_data))

        tensor_data = torch.from_numpy(train_data.astype('float32')).to(self._device)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        data_dim = self.transformer.output_dimensions
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim,self.transformer).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim, self.transformer,self.encoder.positional_encodings).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), 
            lr=self.lr,
            weight_decay=self.l2scale
        )

        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizerAE, 
            T_max=self.epochs,    # 在总epoch数内完成一个余弦周期
            eta_min=1e-7          # 最小学习率
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        # Log model architecture to wandb
        if self.wandb_enabled:
            try:
                import wandb
                wandb.log({
                    "model/encoder_params": sum(p.numel() for p in self.encoder.parameters()),
                    "model/decoder_params": sum(p.numel() for p in self.decoder.parameters()),
                    "model/total_params": sum(p.numel() for p in self.encoder.parameters()) + 
                                         sum(p.numel() for p in self.decoder.parameters()),
                    "data/input_dim": data_dim,
                    "data/train_samples": len(train_data)
                })
            except ImportError:
                print("Wandb not available for logging")

        for i in range(self.epochs):
            recon_losses = []
            kld_losses = []
            
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)  # real torch.size[500,525]
                mu, std, logvar = self.encoder(real)  # [batch_size, num_columns, embedding_dim]
                eps = torch.randn_like(std)   # [batch_size, num_columns, embedding_dim]
                emb = eps * std + mu      # [batch_size, num_columns, embedding_dim]
                rec, sigmas = self.decoder(emb)   # rec dim = data_dim   [batch_size, 525]
                loss_1, loss_2 = _loss_function(
                    rec, real, sigmas, mu, logvar, self.transformer.output_info_list, self.factor
                )
                
                # 添加β调度
                beta = self.get_beta(i)
                loss = loss_1 + beta * loss_2  # 修改这里
                
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
                
                recon_losses.append(loss_1.item())
                kld_losses.append(loss_2.item())

            # Calculate epoch averages
            avg_recon_loss = np.mean(recon_losses)
            avg_kld_loss = np.mean(kld_losses)
            avg_total_loss = avg_recon_loss + avg_kld_loss

            # Log to wandb
            if self.wandb_enabled:
                try:
                    import wandb
                    wandb.log({
                        "train/epoch": i,
                        "train/reconstruction_loss": avg_recon_loss,
                        "train/kld_loss": avg_kld_loss,
                        "train/total_loss": avg_total_loss,
                        "train/sigma_mean": self.decoder.sigma.data.mean().item(),
                        "train/sigma_std": self.decoder.sigma.data.std().item(),
                        "train/sigma_min": self.decoder.sigma.data.min().item(),
                        "train/sigma_max": self.decoder.sigma.data.max().item(),
                    })
                except ImportError:
                    pass

            if self.verbose:
                print(f"Epoch {i:03d} | Recon Loss: {avg_recon_loss:.4f} | KLD Loss: {avg_kld_loss:.4f} | Total: {avg_total_loss:.4f}")

            # Save model every 50 epochs
            if (i + 1) % 50 == 0:
                self.save(f"{self.folder}/vae_epoch_{i+1}.pkl")
                
                # Generate and log sample data every 50 epochs
                if self.wandb_enabled and (i + 1) % 100 == 0:
                    try:
                        import wandb
                        with torch.no_grad():
                            sample_data = self.sample(100)  # Generate 100 samples
                            
                            # Log basic statistics
                            wandb.log({
                                f"samples_epoch_{i+1}/mean": sample_data.mean().mean() if hasattr(sample_data, 'mean') else 0,
                                f"samples_epoch_{i+1}/std": sample_data.std().mean() if hasattr(sample_data, 'std') else 0,
                            })
                            
                    except Exception as e:
                        print(f"Failed to log samples: {e}")

            # 更新学习率
            scheduler.step()

        # Save final model
        self.save(f"{self.folder}/vae_final.pkl")
        
        if self.wandb_enabled:
            try:
                import wandb
                wandb.log({"train/final_epoch": self.epochs})
            except ImportError:
                pass

    @random_state
    def sample(self, samples):
        """
        Sample data similar to the training data.
        
        Args:
            samples (int): Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()
        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            # Sample from latent space for each token
            # Shape: [batch_size, total_tokens, embedding_dim]
            mean = torch.zeros(self.batch_size, self.encoder.total_tokens, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """
        Set the `device` to be used ('GPU' or 'CPU).
        """
        self._device = device
        self.decoder.to(self._device)
