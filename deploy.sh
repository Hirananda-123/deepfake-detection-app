#!/bin/bash

# 🚀 Quick Deployment Script for Deepfake Detection Application

echo "🌐 Deepfake Detection - Quick Deployment Setup"
echo "=============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command_exists git; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

if ! command_exists python3; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Get deployment option
echo ""
echo "🎯 Choose your deployment option:"
echo "1) Heroku (Recommended for beginners)"
echo "2) Railway (Modern alternative)"
echo "3) Docker (Local container)"
echo "4) Manual setup instructions"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🟢 Setting up Heroku deployment..."
        
        if ! command_exists heroku; then
            echo "❌ Heroku CLI not found. Please install it from: https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        
        # Copy production requirements
        cp requirements-production.txt requirements.txt
        
        # Initialize git if not already done
        if [ ! -d ".git" ]; then
            git init
        fi
        
        # Add all files
        git add .
        git commit -m "Initial deployment setup"
        
        # Create Heroku app
        read -p "Enter your app name (or press Enter for auto-generated): " app_name
        
        if [ -z "$app_name" ]; then
            heroku create
        else
            heroku create "$app_name"
        fi
        
        # Deploy to Heroku
        git push heroku main
        
        echo "🎉 Deployed to Heroku! Check your app URL above."
        ;;
        
    2)
        echo "🟡 Setting up Railway deployment..."
        
        if ! command_exists railway; then
            echo "📥 Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        
        # Copy production requirements
        cp requirements-production.txt requirements.txt
        
        # Initialize git if not already done
        if [ ! -d ".git" ]; then
            git init
            git add .
            git commit -m "Initial deployment setup"
        fi
        
        railway login
        railway deploy
        
        echo "🎉 Deployed to Railway! Check the URL provided above."
        ;;
        
    3)
        echo "🔵 Setting up Docker deployment..."
        
        if ! command_exists docker; then
            echo "❌ Docker not found. Please install Docker first."
            exit 1
        fi
        
        # Build Docker image
        echo "🔨 Building Docker image..."
        docker build -t deepfake-detector .
        
        # Run container
        echo "🚀 Starting container..."
        docker run -d -p 8000:8000 --name deepfake-app deepfake-detector
        
        echo "🎉 Application running at http://localhost:8000"
        echo "📊 Container status: docker ps"
        echo "📝 View logs: docker logs deepfake-app"
        echo "⏹️ Stop container: docker stop deepfake-app"
        ;;
        
    4)
        echo "📖 Manual Setup Instructions:"
        echo ""
        echo "1. Choose a hosting platform:"
        echo "   - Heroku: https://heroku.com"
        echo "   - Railway: https://railway.app"
        echo "   - Render: https://render.com"
        echo "   - Vercel: https://vercel.com"
        echo ""
        echo "2. Push your code to GitHub:"
        echo "   git init"
        echo "   git add ."
        echo "   git commit -m 'Initial commit'"
        echo "   git remote add origin YOUR_GITHUB_REPO_URL"
        echo "   git push -u origin main"
        echo ""
        echo "3. Connect your GitHub repo to your chosen platform"
        echo "4. Use requirements-production.txt for dependencies"
        echo "5. Set start command: gunicorn api_server:app --bind 0.0.0.0:\$PORT"
        echo ""
        echo "📚 See WEB_DEPLOYMENT_GUIDE.md for detailed instructions"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "✨ Deployment setup complete!"
echo "📖 For more options, check WEB_DEPLOYMENT_GUIDE.md"
