# 🚀 GitHub Upload Preparation Script (Simple Version)
Write-Host "🌟 Preparing Deepfake Detection Project for GitHub..." -ForegroundColor Green

# Create backup folder for large files
Write-Host "💾 Creating backup for large files..." -ForegroundColor Yellow
if (-not (Test-Path "github_backup")) {
    New-Item -ItemType Directory -Name "github_backup" -Force
}

# Move large model files to backup
$modelFiles = Get-ChildItem -Name "*.pkl"
if ($modelFiles.Count -gt 0) {
    Write-Host "📦 Backing up model files (.pkl)..." -ForegroundColor Yellow
    foreach ($file in $modelFiles) {
        Move-Item $file "github_backup\" -Force
        Write-Host "   ✅ Moved: $file" -ForegroundColor Green
    }
} else {
    Write-Host "   ℹ️  No .pkl files found to backup" -ForegroundColor Cyan
}

# Move zip files to backup
$zipFiles = Get-ChildItem -Name "*.zip"
if ($zipFiles.Count -gt 0) {
    Write-Host "📦 Backing up zip files..." -ForegroundColor Yellow
    foreach ($file in $zipFiles) {
        Move-Item $file "github_backup\" -Force
        Write-Host "   ✅ Moved: $file" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "🎉 Project preparation complete!" -ForegroundColor Green
Write-Host "📚 Next steps:" -ForegroundColor Yellow
Write-Host "   1. Read GITHUB_SETUP_GUIDE.md for detailed instructions" -ForegroundColor White
Write-Host "   2. Go to https://github.com and create a new repository" -ForegroundColor White
Write-Host "   3. Upload your project files using the web interface" -ForegroundColor White
