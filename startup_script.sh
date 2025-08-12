#!/bin/bash

# Startup script for nanoGPT-moe-mup
# This script installs all necessary packages from scratch on a machine with CUDA 12.2

echo "====================================="
echo "nanoGPT-moe-mup Environment Setup"
echo "====================================="
echo ""

# Check if running on Windows
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows system. Please use startup_script.ps1 instead."
    exit 1
fi

# Update system packages
echo "Step 1: Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y python3-pip python3-dev git wget curl

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Step 2: Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "Step 2: Conda already installed, skipping..."
fi

# Create conda environment
echo "Step 3: Creating conda environment..."
conda create -n nanogpt python=3.10 -y
conda activate nanogpt

# Install PyTorch with CUDA 12.2 support
echo "Step 4: Installing PyTorch with CUDA 12.2 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "Step 5: Installing core dependencies..."
pip install numpy
pip install tqdm
pip install pandas
pip install matplotlib
pip install seaborn
pip install requests

# Install tokenizers
echo "Step 6: Installing tokenizers..."
pip install tiktoken

# Install data processing libraries
echo "Step 7: Installing data processing libraries..."
pip install datasets  # For Hugging Face datasets

# Install development tools (optional but recommended)
echo "Step 8: Installing development tools..."
pip install jupyter
pip install ipykernel
pip install black
pip install pylint
pip install pytest

# Install monitoring and logging tools
echo "Step 9: Installing monitoring and logging tools..."
pip install wandb
pip install tensorboard

# Install additional utilities
echo "Step 10: Installing additional utilities..."
pip install pyyaml
pip install rich
pip install click

# Verify CUDA installation
echo ""
echo "Step 11: Verifying CUDA and PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 
python3 -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Clone the repository if not already in it
if [ ! -f "train.py" ]; then
    echo ""
    echo "Step 12: Repository files not found. Please ensure you're in the nanoGPT-moe-mup directory."
    echo "You can clone it with: git clone <your-repository-url>"
else
    echo ""
    echo "Step 12: Repository files found."
fi

# Create necessary directories
echo ""
echo "Step 13: Creating necessary directories..."
mkdir -p out
mkdir -p data
mkdir -p logs

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate nanogpt"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
echo "For distributed training with multiple GPUs:"
echo "  torchrun --standalone --nproc_per_node=<num_gpus> train.py"
echo ""