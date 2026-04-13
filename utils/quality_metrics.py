import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp, spearmanr, cramervonmises_2samp
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

def compute_wasserstein_distance(real, synthetic):
    return wasserstein_distance(real, synthetic)

def compute_ks_test(real, synthetic):
    return ks_2samp(real, synthetic).statistic

def compute_spearman_autocorrelation(data):
    return spearmanr(data[:-1], data[1:]).correlation

def compute_mmd(real, synthetic, kernel='linear'):
    
    # Reshape the data to make it 2D
    real = real.reshape(-1, 1)
    synthetic = synthetic.reshape(-1, 1)
    
    real_real = euclidean_distances(real, real)
    real_synthetic = euclidean_distances(real, synthetic)
    synthetic_synthetic = euclidean_distances(synthetic, synthetic)
    
    if kernel == 'linear':
        return np.mean(real_real) + np.mean(synthetic_synthetic) - 2 * np.mean(real_synthetic)
    else:
        raise ValueError("Unknown kernel")

def sample_time_series_data(data, sequence_length, num_samples):
    """
    Samples time-series data while preserving the sequence structure.
    sequence_length: length of the known sequence in the time-series
    num_samples: number of sequence samples to return
    """
    # Calculate the number of possible sequences
    num_sequences = len(data) - sequence_length + 1

    # Randomly choose start points for sequences
    start_points = np.random.choice(num_sequences, num_samples, replace=False)

    # Collect the sampled sequences
    sampled_data = np.array([data.iloc[start:start + sequence_length] for start in start_points])

    return sampled_data

def compute_mmd_time_series(real, synthetic, kernel='linear', sequence_length=100, num_samples=10):
    # Sample the time-series data
    real_sampled = sample_time_series_data(real, sequence_length, num_samples)
    synthetic_sampled = sample_time_series_data(synthetic, sequence_length, num_samples)
    
    # Flatten the samples for distance calculation
    real_flat = real_sampled.reshape(-1, 1)
    synthetic_flat = synthetic_sampled.reshape(-1, 1)
    
    # Calculate distances
    real_real_dist = np.mean([euclidean_distances(real_batch, real_batch) for real_batch in np.array_split(real_flat, num_samples)])
    real_synthetic_dist = np.mean([euclidean_distances(real_batch, synthetic_batch) for real_batch, synthetic_batch in zip(np.array_split(real_flat, num_samples), np.array_split(synthetic_flat, num_samples))])
    synthetic_synthetic_dist = np.mean([euclidean_distances(synthetic_batch, synthetic_batch) for synthetic_batch in np.array_split(synthetic_flat, num_samples)])
    
    if kernel == 'linear':
        return np.mean(real_real_dist) + np.mean(synthetic_synthetic_dist) - 2 * np.mean(real_synthetic_dist)
    else:
        raise ValueError("Unknown kernel")


def compute_cramervonmises(real, synthetic):
    return cramervonmises_2samp(real, synthetic).statistic

def check_input_shape(real, synthetic):
    if real.ndim != synthetic.ndim:
        raise ValueError("Real and synthetic data dimensions do not match.")
    if real.ndim > 2:
        raise ValueError("Data with more than 2 dimensions not supported.")
    if real.shape[1] != synthetic.shape[1]:
        raise ValueError("Number of features in real and synthetic data do not match.")

# def print_metrics(metrics):
#     num_features = len(metrics) // 5  # Assuming 5 tests are performed
#     for feature in range(num_features):
#         print(f"------- Feature {feature} -----------")
#         for key, value in metrics.items():
#             if f"Feature {feature}" in key:
#                 print(f"{key}: {value:.3f}")
#         print()  # Newline after each feature's metrics

def print_metrics(metrics, sensor_cols):
    for feature in sensor_cols:
        print(f"------- Feature {feature} -----------")
        for key, value in metrics.items():
            if f"Feature {feature}" in key:
                print(f"{key}: {value:.3f}")
        print()  # Newline after each feature's metrics

def print_metrics_to_file(metrics, sensor_cols):
    s = ""
    for feature in sensor_cols:
        s += f"------- Feature {feature} -----------\n"
        for key, value in metrics.items():
            if f"Feature {feature}" in key:
                s += f"{key}: {value:.3f}\n"
        s += "\n\n"  # Newline after each feature's metrics
    return s


def generate_demo_data():
    np.random.seed(33)  # for reproducibility
    num_features = 4
    real_data = np.random.randn(1000, num_features)  # multivariate time-series with 3 features and 1000 samples
    synthetic_data = real_data + 0.5 * np.random.randn(1000, num_features)  # synthetic data created by adding some noise
    
    return real_data, synthetic_data

def evaluate_synthetic_data_univariate(real, synthetic):
    
    # Check input shapes for potential mismatch
    check_input_shape(real, synthetic)
    
    metrics = {
        'Wasserstein Distance': compute_wasserstein_distance(real, synthetic),
        'MMD': compute_mmd(real, synthetic),
        'KS Test Statistic': compute_ks_test(real, synthetic),
        'Cramér-von Mises Statistic': compute_cramervonmises(real, synthetic),
        'Real Data Spearman Autocorrelation': compute_spearman_autocorrelation(real),
        'Synthetic Data Spearman Autocorrelation': compute_spearman_autocorrelation(synthetic)      
    }
    
    return metrics

def evaluate_synthetic_data_multivariate(real, synthetic, sensor_cols, indiv_seq_length):
    print("\nEntering evaluate_synthetic_data_multivariate() function...\n")
    
    # Check input shapes for potential mismatch
    check_input_shape(real, synthetic)
    
    num_features = real.shape[1]
    metrics = {}
    
    # for feature in range(num_features): # List-based
        # list type for 'real' and 'synthetic'
        # real_feature = real[:, feature]
        # synthetic_feature = synthetic[:, feature] 
    
    for feature in tqdm(sensor_cols, desc="evaluate_synthetic_data_multivariate...", ascii=False, ncols=75):
        # dataframe type for 'real' and 'synthetic'
        real_feature = real[feature]
        synthetic_feature = synthetic[feature]
        
        metrics[f'Wasserstein Distance Feature {feature}'] = compute_wasserstein_distance(real_feature, synthetic_feature)
        metrics[f'KS Test Statistic Feature {feature}'] = compute_ks_test(real_feature, synthetic_feature)
        metrics[f'Real Data Spearman Autocorrelation Feature {feature}'] = compute_spearman_autocorrelation(real_feature)
        metrics[f'Synthetic Data Spearman Autocorrelation Feature {feature}'] = compute_spearman_autocorrelation(synthetic_feature)
        # metrics[f'MMD Feature {feature}'] = compute_mmd_time_series(real_feature, synthetic_feature, sequence_length=indiv_seq_length, num_samples=200)
        metrics[f'Cramér-von Mises Statistic Feature {feature}'] = compute_cramervonmises(real_feature, synthetic_feature)
    
    print("\nExiting  evaluate_synthetic_data_multivariate() function...\n")
    return metrics

