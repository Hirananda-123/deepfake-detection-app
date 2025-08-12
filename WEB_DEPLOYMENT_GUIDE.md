# 🌐 Web Deployment Guide - Deepfake Detection Application

## 🎯 Overview
This guide provides multiple options to deploy your deepfake detection application as a live website accessible from anywhere on the internet.

## 📋 Deployment Options

### 🟢 Option 1: Heroku (Recommended for Beginners)
**Free tier available • Easy setup • Automatic scaling**

#### Prerequisites
- Heroku account (free at heroku.com)
- Git installed
- Heroku CLI installed

#### Step-by-Step Deployment

1. **Install Heroku CLI**
```bash
# Download from: https://devcenter.heroku.com/articles/heroku-cli
# Verify installation
heroku --version
```

2. **Login to Heroku**
```bash
heroku login
```

3. **Prepare Application for Heroku**
Create `Procfile` (no extension):
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
```

Create `runtime.txt`:
```
python-3.11.7
```

Update `requirements.txt` to include Heroku dependencies:
```
Flask==3.0.0
gunicorn==21.2.0
flask-cors==4.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
scikit-learn==1.3.2
Pillow==10.1.0
tensorflow-cpu==2.13.0
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
```

4. **Initialize Git and Deploy**
```bash
# Copy the production requirements
cp requirements-production.txt requirements.txt

# Initialize git repository
git init
git add .
git commit -m "Initial deployment"

# Create Heroku app
heroku create your-deepfake-detector

# Deploy to Heroku
git push heroku main
```

5. **Access Your Live Website**
```
https://your-deepfake-detector.herokuapp.com
```

Your website will have:
- **Landing Page** (`/`) - Professional introduction with features
- **Main App** (`/app`) - The deepfake detection tool
- **API Endpoints** (`/api/predict`, `/api/status`) - Backend functionality

---

### 🟡 Option 2: Railway.app (Modern Alternative)
**Free tier • GitHub integration • Simple deployment**

#### Steps:
1. Push your code to GitHub
2. Visit railway.app and connect your GitHub account
3. Select your repository
4. Add environment variables if needed
5. Railway automatically detects Python and deploys

**Live URL**: `https://your-app-name.railway.app`

---

### 🟠 Option 3: Render.com (Free Static + Backend)
**Free tier • Easy setup • Good performance**

#### For Backend API:
1. Create account at render.com
2. Connect GitHub repository
3. Create new "Web Service"
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn api_server:app --bind 0.0.0.0:$PORT`

#### For Frontend:
1. Create "Static Site" service
2. Point to `/frontend` directory
3. Set build command: `cp -r . dist/`

---

### 🔵 Option 4: Vercel (Frontend) + Railway (Backend)
**Excellent performance • Global CDN • Free tier**

#### Frontend on Vercel:
1. Install Vercel CLI: `npm i -g vercel`
2. In `/frontend` directory: `vercel`
3. Follow prompts for deployment

#### Backend on Railway:
1. Follow Railway steps above for API
2. Update frontend API URLs to Railway backend

---

### 🟣 Option 5: AWS (Production-Ready)
**Enterprise-grade • Scalable • Pay-as-you-go**

#### Using AWS Elastic Beanstalk:
1. Install AWS CLI and EB CLI
2. Initialize Elastic Beanstalk:
```bash
eb init
eb create deepfake-detector-env
eb deploy
```

#### Using AWS Lambda + API Gateway:
- Serverless deployment
- Auto-scaling
- Pay per request

---

## 🛠️ Pre-Deployment Preparation

### 1. Update API Server for Production
```python
# api_server.py modifications
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get port from environment variable (for Heroku)
port = int(os.environ.get('PORT', 8000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 2. Environment Variables
Create `.env` file for sensitive data:
```
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MODEL_PATH=./advanced_deepfake_model.pkl
```

### 3. Static File Serving
Update your Flask app to serve frontend files:
```python
@app.route('/')
def index():
    return send_from_directory('frontend', 'modern-index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('frontend', path)
```

## 🔧 Configuration Files

### Dockerfile (for containerized deployment)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "api_server:app", "--bind", "0.0.0.0:8000", "--workers", "1", "--timeout", "120"]
```

### docker-compose.yml
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./temp_uploads:/app/temp_uploads
```

## 🚀 Quick Deployment Commands

### Heroku One-Liner
```bash
# After setting up Procfile and requirements.txt
heroku create your-app-name && git add . && git commit -m "Deploy" && git push heroku main
```

### Railway Deployment
```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway deploy
```

### Docker Deployment
```bash
# Build and run locally
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector

# Deploy to any cloud that supports Docker
```

## 📱 Domain & SSL Setup

### Custom Domain
1. **Purchase domain** (GoDaddy, Namecheap, etc.)
2. **Configure DNS** to point to your hosting provider
3. **Add domain** in your hosting platform settings

### SSL Certificate
Most platforms provide free SSL:
- **Heroku**: Automatic with custom domains
- **Vercel**: Automatic
- **Railway**: Automatic
- **AWS**: Use Certificate Manager

## 🔍 Performance Optimization

### 1. Model Optimization
```python
# Compress model file
import joblib
model = joblib.load('model.pkl')
joblib.dump(model, 'model_compressed.pkl', compress=3)
```

### 2. Image Processing
```python
# Optimize image handling
from PIL import Image

def optimize_image(image_path, max_size=(800, 800)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        return img
```

### 3. Caching
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def analyze_image(image_hash):
    # Your analysis logic
    pass
```

## 🔒 Security Best Practices

### 1. File Upload Security
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
```

### 2. Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Your prediction logic
    pass
```

### 3. Environment Variables
```python
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')
```

## 📊 Monitoring & Analytics

### 1. Application Monitoring
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Prediction request from {request.remote_addr}")
    # Your logic here
```

### 2. Performance Tracking
```python
import time

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    diff = time.time() - g.start_time
    logger.info(f"Request took {diff:.2f} seconds")
    return response
```

## 🎯 Recommended Deployment Strategy

### For Development/Demo:
1. **Heroku** - Quick and free
2. **Railway** - Modern alternative

### For Production:
1. **AWS Elastic Beanstalk** - Scalable and reliable
2. **Google Cloud Run** - Serverless containers
3. **DigitalOcean App Platform** - Developer-friendly

### For High Traffic:
1. **AWS ECS/EKS** - Container orchestration
2. **Google Kubernetes Engine** - Advanced scaling
3. **Azure Container Instances** - Microsoft ecosystem

## 🆘 Troubleshooting

### Common Deployment Issues:

**Build Failures:**
- Check Python version compatibility
- Verify requirements.txt is complete
- Ensure all files are committed to git

**Memory Issues:**
- Use `tensorflow-cpu` instead of `tensorflow`
- Optimize model size
- Increase worker memory limits

**Timeout Issues:**
- Increase timeout settings
- Optimize image processing
- Use background tasks for heavy operations

**CORS Issues:**
```python
from flask_cors import CORS
CORS(app, origins=['https://yourdomain.com'])
```

---

## 🎉 Go Live Checklist

- [ ] Code pushed to git repository
- [ ] Dependencies listed in requirements.txt
- [ ] Environment variables configured
- [ ] Production server settings applied
- [ ] SSL certificate enabled
- [ ] Custom domain configured (optional)
- [ ] Monitoring and logging set up
- [ ] Performance optimization applied
- [ ] Security measures implemented
- [ ] Backup strategy in place

**Your deepfake detection application is ready for the world! 🚀**

---

*Created: August 12, 2025*  
*Last Updated: August 12, 2025*  
*Version: 2.0*
