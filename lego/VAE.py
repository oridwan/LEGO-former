"""
VAE module.
"""
import os
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
    """

    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, input_):
        """Encode the passed `input_`."""
        feature = self.seq(input_)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    """Decoder for the TVAE.

    Args:
        embedding_dim (int): Size of the input vector.
        decompress_dims (tuple or list of ints): Size of each hidden layer.
        data_dim (int): Dimensions of the data.
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, input_):
        """Decode the passed `input_`."""
        return self.seq(input_), self.sigma

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
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
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
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=False,
        type='discrete',
        folder='LEGO-VAE',
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)
        #store encoded features for last step
        self.full_encoded_mu = []
        self.full_encoded_std = []
        self.type = type
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
        print("Train Data", np.shape(train_data), '\n', train_data[:10])
        print("Transformed Data shape", np.shape(train_data))

        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        self.encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            #reset full encoded features
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = self.encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })

            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(loss=loss.detach().cpu().item())
                )
            if (i + 1) % 50 == 0:  
                # Save model
                model_path = f'{self.model_folder}/VAE_{self.type}_model_checkpoint_epoch_{i+1}.pkl'
                self.save(model_path)
                print(f"Model saved at epoch {i+1} to {model_path}")
                
                # Generate 100k samples
                samples_path = f'{self.samples_folder}/VAE_{self.type}_samples_epoch_{i+1}.csv'
                samples = self.sample(500000)
                
                # Save samples (assuming they're a DataFrame or can be converted to one)
                if isinstance(samples, np.ndarray):
                    pd.DataFrame(samples).to_csv(samples_path, index=False)
                else:
                    samples.to_csv(samples_path, index=False)
                print(f"Generated 100k samples at epoch {i+1}, saved to {samples_path}")

        self.plot_losses(self.loss_values, filename='epoch_loss_plot.png')

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
            mean = torch.zeros(self.batch_size, self.embedding_dim)
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
