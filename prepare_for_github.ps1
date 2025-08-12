# 🚀 GitHub Upload Preparation Script
# This script prepares your project for GitHub upload

Write-Host "🌟 Preparing Deepfake Detection Project for GitHub..." -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# Check current directory
$currentDir = Get-Location
Write-Host "📁 Current directory: $currentDir" -ForegroundColor Blue

# Create backup folder for large files
Write-Host "💾 Creating backup for large files..." -ForegroundColor Yellow
if (-not (Test-Path "github_backup")) {
    New-Item -ItemType Directory -Name "github_backup" -Force
}

# Move large model files to backup
$modelFiles = Get-ChildItem -Name "*.pkl"
if ($modelFiles) {
    Write-Host "📦 Backing up model files (.pkl)..." -ForegroundColor Yellow
    foreach ($file in $modelFiles) {
        Move-Item $file "github_backup\" -Force
        Write-Host "   ✅ Moved: $file" -ForegroundColor Green
    }
} else {
    Write-Host "   ℹ️  No .pkl files found to backup" -ForegroundColor Cyan
}

# Clean up zip files
$zipFiles = Get-ChildItem -Name "*.zip"
if ($zipFiles) {
    Write-Host "📦 Backing up zip files..." -ForegroundColor Yellow
    foreach ($file in $zipFiles) {
        Move-Item $file "github_backup\" -Force
        Write-Host "   ✅ Moved: $file" -ForegroundColor Green
    }
}

# Clean up cache directories
Write-Host "🧹 Cleaning up cache directories..." -ForegroundColor Yellow
if (Test-Path "__pycache__") {
    Remove-Item "__pycache__" -Recurse -Force
    Write-Host "   ✅ Removed: __pycache__" -ForegroundColor Green
}

if (Test-Path ".venv\**\__pycache__") {
    Get-ChildItem -Path ".venv" -Recurse -Name "__pycache__" | ForEach-Object {
        Remove-Item $_ -Recurse -Force
    }
    Write-Host "   ✅ Cleaned: .venv cache files" -ForegroundColor Green
}

# Check file sizes
Write-Host "📊 Checking file sizes..." -ForegroundColor Yellow
$largeFiles = Get-ChildItem -Recurse | Where-Object { 
    $_.Length -gt 100MB -and 
    $_.Name -notlike "*.venv*" -and 
    $_.Name -notlike "*github_backup*"
}

if ($largeFiles) {
    Write-Host "⚠️  Large files found (>100MB):" -ForegroundColor Red
    foreach ($file in $largeFiles) {
        $sizeMB = [Math]::Round($file.Length / 1MB, 2)
        Write-Host "   📄 $($file.Name): ${sizeMB}MB" -ForegroundColor Red
    }
    Write-Host "   💡 Consider using Git LFS for these files" -ForegroundColor Yellow
} else {
    Write-Host "   ✅ No large files found" -ForegroundColor Green
}

# Count files to upload
$filesToUpload = Get-ChildItem -Recurse -File | Where-Object { 
    $_.FullName -notlike "*\.venv\*" -and 
    $_.FullName -notlike "*\github_backup\*" -and
    $_.FullName -notlike "*\__pycache__\*"
}

Write-Host "📈 Project Statistics:" -ForegroundColor Blue
Write-Host "   📁 Total files to upload: $($filesToUpload.Count)" -ForegroundColor Cyan
Write-Host "   📊 Total size: $([Math]::Round(($filesToUpload | Measure-Object Length -Sum).Sum / 1MB, 2))MB" -ForegroundColor Cyan

# Show key files status
Write-Host "🔍 Key files check:" -ForegroundColor Blue
$keyFiles = @("README.md", "requirements.txt", "app.py", "Procfile", ".gitignore")
foreach ($file in $keyFiles) {
    if (Test-Path $file) {
        Write-Host "   ✅ $file" -ForegroundColor Green
    } else {
        Write-Host "   ❌ $file (missing)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 Project preparation complete!" -ForegroundColor Green
Write-Host "📚 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Read GITHUB_SETUP_GUIDE.md for detailed instructions" -ForegroundColor White
Write-Host "   2. Choose upload method (GitHub Desktop, Web, or Git CLI)" -ForegroundColor White
Write-Host "   3. Create repository on GitHub" -ForegroundColor White
Write-Host "   4. Upload your files" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tip: Use Method 3 (Web Interface) if you don't have Git installed" -ForegroundColor Cyan
Write-Host "🔗 GitHub: https://github.com" -ForegroundColor Blue

# Ask if user wants to open GitHub
$openGitHub = Read-Host "🌐 Open GitHub in your browser? (y/n)"
if ($openGitHub -eq "y" -or $openGitHub -eq "Y") {
    Start-Process "https://github.com"
}

pause
