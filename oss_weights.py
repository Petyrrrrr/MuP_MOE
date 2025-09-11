import torch
import numpy as np

def load_router_weights(file_path='/home/ubuntu/MuP_MOE/gpt_oss_router/router_weights.pt'):
    """
    Load router weights and return as a list of 24 matrices.
    
    Returns:
        list: List of 24 numpy arrays, each of shape (32, 2880)
    """
    # Load the weights
    router_weights = torch.load(file_path, map_location='cpu')
    
    # Convert to list of numpy arrays
    matrices = []
    for i in range(24):
        layer_key = f'layer_{i}'
        matrix = router_weights[layer_key].numpy()
        matrices.append(matrix)
    
    return matrices

def print_matrices_info(matrices):
    """Print information about each matrix."""
    for i, matrix in enumerate(matrices):
        print(f"Layer {i}: shape {matrix.shape}, dtype {matrix.dtype}")
        print(f"  Min: {matrix.min():.6f}, Max: {matrix.max():.6f}, Mean: {matrix.mean():.6f}")
        print(f"  Std: {matrix.std():.6f}")
        print()

def save_matrices_separately(matrices, output_dir='/home/ubuntu/MuP_MOE/router_matrices'):
    """Save each matrix as a separate .npy file."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, matrix in enumerate(matrices):
        output_path = os.path.join(output_dir, f'layer_{i}.npy')
        np.save(output_path, matrix)
        print(f"Saved layer {i} to {output_path}")

if __name__ == "__main__":
    # Load the weights as a list of matrices
    matrices = load_router_weights()
    
    print(f"Loaded {len(matrices)} matrices")
    print(f"Each matrix has shape: {matrices[0].shape}")
    print("\nDetailed information for each layer:")
    print("="*50)
    
    # Print information about each matrix
    print_matrices_info(matrices)
    
    # Example: Access specific matrix
    print("\nExample usage:")
    print("matrices[0] is the router weights for layer 0")
    print(f"matrices[0].shape = {matrices[0].shape}")
    print(f"matrices[0][0, :10] = {matrices[0][0, :10]}")  # First 10 weights for expert 0 in layer 0
    
    # Optional: Save matrices separately
    # Uncomment the following line to save each matrix as a separate file
    # save_matrices_separately(matrices)