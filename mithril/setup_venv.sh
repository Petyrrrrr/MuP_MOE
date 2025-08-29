#!/bin/bash

# Create and setup Python virtual environment for MF_MOE_toy project

echo "Setting up Python virtual environment..."

# Install Python development headers (required for PyTorch Triton compilation)
echo "Installing Python development headers..."
sudo apt update
sudo apt install -y python3.12-dev build-essential

# Check if venv already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old venv..."
    rm -rf venv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✓ Virtual environment setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
echo "Your Python interpreter is now at: $(which python)"
echo "Your pip is now at: $(which pip)"