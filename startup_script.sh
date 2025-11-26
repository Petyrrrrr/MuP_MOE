#!/bin/bash

# Run the mithril setup script to create virtual environment and install dependencies
echo "Step 1: Setting up Python virtual environment..."
if [ -f "mithril/setup_venv.sh" ]; then
    chmod +x mithril/setup_venv.sh
    bash mithril/setup_venv.sh
    
    # Activate the virtual environment for the rest of this script
    source venv/bin/activate
else
    echo "Error: mithril/setup_venv.sh not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

if [ -f "mithril/install-node.sh" ]; then
    chmod +x mithril/install-node.sh
    bash mithril/install-node.sh
else
    echo "Error: mithril/install-node.sh not found!"
    echo "Please ensure you're in the correct directory."
    exit 1
fi

# Verify CUDA installation
echo ""
echo "Step 2: Verifying CUDA and PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')" 
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"

# Prepare Shakespeare dataset
echo ""
echo "Step 3: Preparing Shakespeare character-level dataset..."
if [ -f "data/shakespeare_char/prepare.py" ]; then
    python data/shakespeare_char/prepare.py
    echo "Shakespeare dataset prepared successfully!"
else
    echo "Warning: data/shakespeare_char/prepare.py not found. Skipping dataset preparation."
    echo "You may need to prepare the data manually by running:"
    echo "  python data/shakespeare_char/prepare.py"
fi

echo ""
echo "====================================="
echo "Setup Complete!"
echo "====================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  python train.py"
echo ""
echo "For distributed training with multiple GPUs:"
echo "  torchrun --standalone --nproc_per_node=<num_gpus> train.py"
echo ""

git config --global user.email "jtz2003@qq.com"
git config --global user.name "Petyrrrrr"
bash wandb_start.sh

mkdir std_out