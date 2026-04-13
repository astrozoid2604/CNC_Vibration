import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.stats import zscore
from torch.utils.data import DataLoader, TensorDataset

# Simulated GAN components
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, data_length),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(data_length, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def spectral_loss(fft_real, fft_generated, weight_matrix):
    """
    Calculate the spectral loss between FFTs of real and generated data.
    Args:
    - fft_real (torch.Tensor): FFT of real data samples
    - fft_generated (torch.Tensor): FFT of generated data samples
    - weight_matrix (torch.Tensor): Dynamic weights for frequency components
    Returns:
    - torch.Tensor: Calculated spectral loss
    """
    loss = torch.zeros(fft_real.size(0), device=fft_real.device)

    # Eqn (11) - Calculates the squared Euclidean distance for a single frequency point between the real and generated (fake) samples 
    for i in range(fft_real.size(1)):
        loss += weight_matrix[i] * torch.abs(fft_real[:, i] - fft_generated[:, i]).pow(2)
    
    # Eqn (12) - Based on the weighted squared difference results from Eqn(11), take the mean across entire spectrum.
    return loss.mean()

def update_weight_matrix(fft_real, fft_generated, weight_matrix, flexibility_scaling_factor):
    """
    Update the weight matrix based on FFTs of real and generated data.
    Args:
    - fft_real (torch.Tensor): FFT of real data samples
    - fft_generated (torch.Tensor): FFT of generated data samples
    - weight_matrix (torch.Tensor): Current weight matrix
    - flexibility_scaling_factor (float): Scaling factor for weight updates
    Returns:
    - torch.Tensor: Updated weight matrix
    """
    error = torch.abs(fft_real - fft_generated).mean(dim=0)
    weight_matrix.mul_(flexibility_scaling_factor).add_(error)
    weight_matrix.div_(weight_matrix.sum())
    return weight_matrix

def update_weight_matrix_paper_interpretation(fft_real, fft_generated, weight_matrix, alpha):
    """
    Update the weight matrix based on the paper's methodology.
    Args:
    - fft_real (torch.Tensor): FFT of real data samples
    - fft_generated (torch.Tensor): FFT of generated data samples
    - weight_matrix (torch.Tensor): Current weight matrix
    - alpha (float): Alpha parameter from the paper
    Returns:
    - torch.Tensor: Updated weight matrix
    """
    # Calculate the frequency distance based on Eq. (11) and (12)
    freq_distance = torch.abs(fft_real - fft_generated).pow(2)
    freq_distance_mean = torch.mean(freq_distance, dim=0)

    # Update the weight matrix based on the paper's methodology (Eq. 13)
    # Note: This is an interpretation based on my understanding of the paper's description
    updated_weights = weight_matrix * alpha + freq_distance_mean
    updated_weights_normalized = updated_weights / torch.sum(updated_weights)

    return updated_weights_normalized


# ---------------- Experiment Setup 
# Parameters
sampling_rate               = 25000  # Sampling rate of 25kHz.
num_epochs                  = 10  # Example number of epochs
flexibility_scaling_factor  = 0.1  # Scaling factor for weight matrix update
batch_size                  = 32  # Batch size
data_length                 = 1024  # Sample data length -> replaced with partition length for DGAN?
ckpt_save_interval          = 3
discrim_training_ratio      = 4
# ----------------

# ---------------- Manipulate data
# Sample initialization of data and parameters for demonstration purposes
# Replace with actual data loading in practice
real_data_np = np.random.randn(100, data_length)  # Generating NumPy array of real data
real_data_normalized = zscore(real_data_np, axis=None)  # Normalizing
real_data = torch.tensor(real_data_normalized, dtype=torch.float32)  # Converting to torch tensor
# generated_data = torch.randn(100, data_length)  # 100 samples of generated data

# DataLoader for batch processing
dataset = TensorDataset(real_data) #, generated_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# ----------------

# ---------------- Model Related Parameters
# Initialize the weight matrix with equal weights
dynamic_weight_matrix = torch.ones(data_length)  # Initialize weight matrix

# Initialize Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Binary Cross Entropy Loss - Example only
adversarial_loss = nn.BCELoss()
# ----------------

# ----------------
# Training loop
for epoch in range(num_epochs):
    for data_real, in dataloader:
        # Adversarial ground truths
        valid = torch.ones(data_real.size(0), 1)
        fake = torch.zeros(data_real.size(0), 1)

        # -----------------
        # Train Discriminator Round
        for _ in range(discrim_training_ratio):  # Train discriminator 4 times
            d_optimizer.zero_grad()

            # Real data
            real_loss = adversarial_loss(discriminator(data_real), valid)
            
            # -------------- DEMO: Fake Data Generation per batch
            # Generate a batch of fake data
            z = torch.randn(data_real.size(0), 100)
            data_generated = generator(z)

            # Standardize and normalize generated data
            generated_data_np = data_generated.detach().numpy()
            standardized_generated_data = zscore(generated_data_np, axis=1)
            normalized_generated_data = 2 * (standardized_generated_data - np.min(standardized_generated_data, axis=1, keepdims=True)) / \
                                        (np.max(standardized_generated_data, axis=1, keepdims=True) - np.min(standardized_generated_data, axis=1, keepdims=True)) - 1
            data_generated = torch.tensor(normalized_generated_data, dtype=torch.float32)
        # --------------                                      
        
        # Fake data
        fake_loss = adversarial_loss(discriminator(data_generated.detach()), fake)

        # Total Discriminator loss and backprop
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        # -----------------

        # -----------------
        # Train Generator Round
        g_optimizer.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(data_generated), valid)

        # Apply FFT to both real and generated data
        fft_real = torch.fft.fft(data_real)
        fft_generated = torch.fft.fft(data_generated)

        # Spectral loss calculation
        spec_loss = spectral_loss(fft_real, fft_generated, dynamic_weight_matrix)

        # Combined Generator loss and backprop
        combined_loss = g_loss + spec_loss # To include other losses for DGAN model.
        combined_loss.backward()
        g_optimizer.step()
        # -----------------

        # -----------------
        # Update dynamic weight matrix
        dynamic_weight_matrix = update_weight_matrix(fft_real, fft_generated, dynamic_weight_matrix, flexibility_scaling_factor=1)
        # -----------------

        # Print losses and epoch information
        print(f"Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Spectral Loss: {spec_loss.item()}")

        # Include rest of the training code (save # Save model checkpoints periodically
        # Checkpoint saving         
        # Tensorboard model save 
        # Adjust learning rate 



