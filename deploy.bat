@echo off
echo 🌐 Deepfake Detection - Quick Deployment Setup
echo ==============================================

echo 📋 Checking prerequisites...

where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Prerequisites check passed!
echo.

echo 🎯 Choose your deployment option:
echo 1) Heroku (Recommended for beginners)
echo 2) Railway (Modern alternative) 
echo 3) Docker (Local container)
echo 4) Manual setup instructions

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo 🟢 Setting up Heroku deployment...
    
    where heroku >nul 2>nul
    if %errorlevel% neq 0 (
        echo ❌ Heroku CLI not found. Please install it from: https://devcenter.heroku.com/articles/heroku-cli
        pause
        exit /b 1
    )
    
    rem Copy production requirements
    copy requirements-production.txt requirements.txt
    
    rem Initialize git if not already done
    if not exist ".git" (
        git init
    )
    
    rem Add all files
    git add .
    git commit -m "Initial deployment setup"
    
    rem Create Heroku app
    set /p app_name="Enter your app name (or press Enter for auto-generated): "
    
    if "%app_name%"=="" (
        heroku create
    ) else (
        heroku create %app_name%
    )
    
    rem Deploy to Heroku
    git push heroku main
    
    echo 🎉 Deployed to Heroku! Check your app URL above.
    
) else if "%choice%"=="2" (
    echo 🟡 Setting up Railway deployment...
    
    where railway >nul 2>nul
    if %errorlevel% neq 0 (
        echo 📥 Installing Railway CLI...
        npm install -g @railway/cli
    )
    
    rem Copy production requirements
    copy requirements-production.txt requirements.txt
    
    rem Initialize git if not already done
    if not exist ".git" (
        git init
        git add .
        git commit -m "Initial deployment setup"
    )
    
    railway login
    railway deploy
    
    echo 🎉 Deployed to Railway! Check the URL provided above.
    
) else if "%choice%"=="3" (
    echo 🔵 Setting up Docker deployment...
    
    where docker >nul 2>nul
    if %errorlevel% neq 0 (
        echo ❌ Docker not found. Please install Docker first.
        pause
        exit /b 1
    )
    
    rem Build Docker image
    echo 🔨 Building Docker image...
    docker build -t deepfake-detector .
    
    rem Run container
    echo 🚀 Starting container...
    docker run -d -p 8000:8000 --name deepfake-app deepfake-detector
    
    echo 🎉 Application running at http://localhost:8000
    echo 📊 Container status: docker ps
    echo 📝 View logs: docker logs deepfake-app
    echo ⏹️ Stop container: docker stop deepfake-app
    
) else if "%choice%"=="4" (
    echo 📖 Manual Setup Instructions:
    echo.
    echo 1. Choose a hosting platform:
    echo    - Heroku: https://heroku.com
    echo    - Railway: https://railway.app
    echo    - Render: https://render.com
    echo    - Vercel: https://vercel.com
    echo.
    echo 2. Push your code to GitHub:
    echo    git init
    echo    git add .
    echo    git commit -m "Initial commit"
    echo    git remote add origin YOUR_GITHUB_REPO_URL
    echo    git push -u origin main
    echo.
    echo 3. Connect your GitHub repo to your chosen platform
    echo 4. Use requirements-production.txt for dependencies
    echo 5. Set start command: gunicorn app:app --bind 0.0.0.0:$PORT
    echo.
    echo 📚 See WEB_DEPLOYMENT_GUIDE.md for detailed instructions
    
) else (
    echo ❌ Invalid choice. Please run the script again.
    pause
    exit /b 1
)

echo.
echo ✨ Deployment setup complete!
echo 📖 For more options, check WEB_DEPLOYMENT_GUIDE.md
pause
