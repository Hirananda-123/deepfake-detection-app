# PowerShell script to start the Deepfake Detection API Server
Write-Host "Starting Deepfake Detection API Server..." -ForegroundColor Green

# Change to project directory
Set-Location "C:\Users\815863\OneDrive - Cognizant\Desktop\Github Copilot\Deepfake detection"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

# Test environment
Write-Host "Testing environment setup..." -ForegroundColor Yellow
& python test_env.py

# Start Flask server
Write-Host "Starting Flask API server..." -ForegroundColor Green
Write-Host "The server will be available at http://localhost:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
& python api_server.py
