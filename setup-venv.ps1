# Virtual Environment Setup Script for gjsmovie
# Run this script in PowerShell to create and activate a virtual environment

Write-Host "Creating virtual environment..." -ForegroundColor Green

# Create virtual environment
python -m venv .venv

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create virtual environment. Using global Python instead." -ForegroundColor Yellow
    Write-Host "Installing dependencies to global Python..." -ForegroundColor Green
    pip install -r requirements.txt
    exit
}

# Activate virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Green
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Write-Host "Virtual environment setup complete!" -ForegroundColor Green
Write-Host "To activate in the future, run: .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
