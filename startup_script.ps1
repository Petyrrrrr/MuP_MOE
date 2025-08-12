# Startup script for nanoGPT-moe-mup (Windows PowerShell)
# This script installs all necessary packages from scratch on a Windows machine with CUDA 12.2

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "nanoGPT-moe-mup Environment Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "Step 1: Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Python not found. Please install Python 3.10 or later from https://www.python.org/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Red
    exit 1
}

# Check if pip is installed
Write-Host ""
Write-Host "Step 2: Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = python -m pip --version
    Write-Host "pip found: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "pip not found. Installing pip..." -ForegroundColor Yellow
    python -m ensurepip --upgrade
}

# Upgrade pip
Write-Host ""
Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Create virtual environment
Write-Host ""
Write-Host "Step 4: Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists. Removing old environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force venv
}
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Step 5: Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install PyTorch with CUDA 12.2 support
Write-Host ""
Write-Host "Step 6: Installing PyTorch with CUDA 12.2 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
Write-Host ""
Write-Host "Step 7: Installing core dependencies..." -ForegroundColor Yellow
pip install numpy
pip install tqdm
pip install pandas
pip install matplotlib
pip install seaborn
pip install requests

# Install tokenizers
Write-Host ""
Write-Host "Step 8: Installing tokenizers..." -ForegroundColor Yellow
pip install tiktoken

# Install data processing libraries
Write-Host ""
Write-Host "Step 9: Installing data processing libraries..." -ForegroundColor Yellow
pip install datasets  # For Hugging Face datasets

# Install development tools (optional but recommended)
Write-Host ""
Write-Host "Step 10: Installing development tools..." -ForegroundColor Yellow
pip install jupyter
pip install ipykernel
pip install black
pip install pylint
pip install pytest

# Install monitoring and logging tools
Write-Host ""
Write-Host "Step 11: Installing monitoring and logging tools..." -ForegroundColor Yellow
pip install wandb
pip install tensorboard

# Install additional utilities
Write-Host ""
Write-Host "Step 12: Installing additional utilities..." -ForegroundColor Yellow
pip install pyyaml
pip install rich
pip install click

# Verify CUDA installation
Write-Host ""
Write-Host "Step 13: Verifying CUDA and PyTorch installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

# Check if repository files exist
Write-Host ""
Write-Host "Step 14: Checking repository files..." -ForegroundColor Yellow
if (!(Test-Path "train.py")) {
    Write-Host "Repository files not found. Please ensure you're in the nanoGPT-moe-mup directory." -ForegroundColor Red
    Write-Host "You can clone it with: git clone <your-repository-url>" -ForegroundColor Red
} else {
    Write-Host "Repository files found." -ForegroundColor Green
}

# Create necessary directories
Write-Host ""
Write-Host "Step 15: Creating necessary directories..." -ForegroundColor Yellow
if (!(Test-Path "out")) { New-Item -ItemType Directory -Path "out" }
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" }
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" }

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Green
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To start training, run:" -ForegroundColor Green
Write-Host "  python train.py" -ForegroundColor White
Write-Host ""
Write-Host "For distributed training with multiple GPUs:" -ForegroundColor Green
Write-Host "  python -m torch.distributed.run --standalone --nproc_per_node=<num_gpus> train.py" -ForegroundColor White
Write-Host ""