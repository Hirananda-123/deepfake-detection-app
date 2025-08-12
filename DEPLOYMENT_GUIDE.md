# 🎯 Deepfake Detection Application - Deployment Guide

## 📦 Package Contents

This zip file contains a complete deepfake detection web application with the following components:

### 🔧 Core Application Files
- `api_server.py` - Main Flask API server
- `simple_detector.py` - ML-based deepfake detection engine
- `simple_deepfake_model.pkl` - Trained machine learning model
- `launch.py` - One-click application launcher

### 🌐 Frontend Files
- `frontend/index.html` - Main web interface
- `frontend/styles.css` - UI styling
- `frontend/script.js` - Frontend functionality

### 📚 Documentation & Setup
- `README.md` - Main project documentation
- `FRONTEND_GUIDE.md` - Frontend usage instructions
- `DEPLOYMENT_GUIDE.md` - This deployment guide
- `requirements.txt` - Python dependencies

### 🧪 Test Files
- `test_face.jpg` - Sample image for testing

## 🚀 Quick Start Deployment

### Prerequisites
- Python 3.8+ installed
- Windows/Linux/macOS compatible

### 1. Extract the Package
```bash
# Extract the zip file to your desired directory
unzip deepfake_detection_app.zip
cd deepfake_detection_app
```

### 2. Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
# Option 1: Use the launcher (recommended)
python launch.py

# Option 2: Start API server directly
python api_server.py
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000/frontend
- **API Endpoint**: http://localhost:8000/predict
- **Health Check**: http://localhost:8000/status

## 🎮 How to Use

### Upload and Analyze Images
1. Open http://localhost:8000/frontend in your browser
2. **Upload an image** by either:
   - Dragging and dropping an image file
   - Clicking "Choose File" to browse
3. **Click "Analyze Image"** after upload
4. **View results** showing if the image is Real or Fake with confidence scores

### Supported Image Formats
- JPG, JPEG, PNG, GIF, BMP, TIFF
- Maximum file size: 16MB

## 🔧 Configuration

### Port Configuration
To change the default port (8000), edit `api_server.py`:
```python
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=YOUR_PORT, debug=False)
```

### Model Settings
The application uses a traditional ML approach with OpenCV for face detection. The model is pre-trained and ready to use.

## 🛠️ API Usage

### Predict Endpoint
```bash
# Test with curl
curl -X POST \
  http://localhost:8000/predict \
  -F "image=@your_image.jpg"
```

### Response Format
```json
{
  "prediction": "REAL" or "FAKE",
  "confidence": 0.85,
  "processing_time": 1.23,
  "face_detected": true,
  "model_type": "simple"
}
```

## 🔍 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port 8000
netstat -ano | findstr :8000
# Kill the process and restart
```

**Module Not Found Errors**
```bash
# Ensure virtual environment is activated
# Reinstall requirements
pip install -r requirements.txt
```

**No Face Detected**
- Ensure the image contains a clear, frontal face
- Try different lighting or angle
- Use the provided test_face.jpg for verification

### System Requirements
- **RAM**: Minimum 2GB, Recommended 4GB+
- **Storage**: 100MB for application + dependencies
- **Python**: Version 3.8 or higher

## 📈 Performance Notes

- **Processing Time**: 1-3 seconds per image
- **Accuracy**: Training data dependent (demo version)
- **Concurrent Users**: Development server (single-threaded)

## 🔒 Security Considerations

- This is a development server - not for production use
- File uploads are temporarily stored and cleaned up
- No authentication implemented (add as needed)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure virtual environment is properly activated

## 🎯 Next Steps for Production

1. **Use a production WSGI server** (gunicorn, uWSGI)
2. **Add authentication and authorization**
3. **Implement file size/type validation**
4. **Add logging and monitoring**
5. **Configure HTTPS**
6. **Add rate limiting**

---

**Created**: August 11, 2025  
**Version**: 1.0  
**License**: MIT  
