# Deepfake Detection System

A comprehensive deepfake detection application using deep learning to identify artificially generated or manipulated images and videos.

## Features

- **Multiple Model Architectures**: Custom CNN, EfficientNetB0, and ResNet50V2
- **Image and Video Detection**: Analyze both static images and video files
- **Face Detection**: Automatic face extraction and analysis
- **Web Interface**: User-friendly Gradio-based web application
- **Batch Processing**: Process multiple files at once
- **High Accuracy**: Trained on DFDC Preview dataset

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API** (for dataset download):
   - Create a Kaggle account and get API credentials
   - Place `kaggle.json` in your home directory under `.kaggle/`

## Quick Start

### 1. Download Dataset
```bash
python "import os.py"
```

### 2. Train a Model
```bash
python train_model.py
```

### 3. Run Web Application
```bash
python web_app.py
```

### 4. Test Setup (Optional)
```bash
python train_model.py test
```

## File Structure

```
deepfake-detection/
├── import os.py              # Dataset download script
├── data_preprocessing.py     # Data preprocessing utilities
├── deepfake_detector.py      # Model architectures and training
├── predictor.py             # Prediction utilities
├── train_model.py           # Main training pipeline
├── web_app.py               # Gradio web interface
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Usage

### Training a Model

1. **Download the dataset**:
   ```bash
   python "import os.py"
   ```

2. **Start training**:
   ```bash
   python train_model.py
   ```

3. **Choose model architecture** when prompted:
   - Custom CNN (lightweight, faster training)
   - EfficientNetB0 (recommended, balanced performance)
   - ResNet50V2 (high accuracy, slower training)

### Making Predictions

#### Using the Web Interface
```bash
python web_app.py
```
Then open http://localhost:7860 in your browser.

#### Using Python Code
```python
from predictor import DeepfakePredictor

# Load trained model
predictor = DeepfakePredictor("efficientnet_best.h5")

# Predict single image
label, confidence = predictor.predict_image("path/to/image.jpg")
print(f"Prediction: {label} (Confidence: {confidence:.2f})")

# Predict video
label, confidence = predictor.predict_video("path/to/video.mp4")
print(f"Video prediction: {label} (Confidence: {confidence:.2f})")
```

#### Batch Processing
```python
# Process all images in a directory
results = predictor.batch_predict_directory("path/to/images/")
```

## Model Architectures

### 1. Custom CNN
- Lightweight architecture
- Good for quick experiments
- Lower computational requirements

### 2. EfficientNetB0 (Recommended)
- Pre-trained on ImageNet
- Excellent balance of accuracy and speed
- Transfer learning with fine-tuning

### 3. ResNet50V2
- Deep residual network
- High accuracy potential
- More computational resources required

## Dataset

The system uses the **DFDC Preview dataset** from Kaggle, which contains:
- Real and manipulated videos
- Diverse scenarios and quality levels
- Metadata with ground truth labels

## Performance Metrics

The system tracks multiple metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Web Interface Features

- **Image Upload**: Drag and drop or browse for images
- **Video Upload**: Support for common video formats
- **Real-time Analysis**: Immediate results with confidence scores
- **User-friendly Interface**: Clean, intuitive design

## Technical Details

### Data Preprocessing
- Automatic face detection using OpenCV
- Image normalization and resizing
- Data augmentation for training
- Train/validation split with stratification

### Model Training
- Adam optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Model checkpointing for best weights
- Comprehensive training history visualization

### Prediction Pipeline
- Face detection and extraction
- Image preprocessing and normalization
- Model inference with confidence scoring
- Result visualization and interpretation

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **GPU Issues**: The system works with both CPU and GPU
   - Check TensorFlow GPU installation if using GPU
   - CPU training is slower but functional

3. **Dataset Not Found**: Ensure the dataset is downloaded
   ```bash
   python "import os.py"
   ```

4. **Model Loading Errors**: Check model file paths and ensure models are trained

### Performance Tips

- Use GPU if available for faster training
- Adjust batch size based on available memory
- Use EfficientNetB0 for best balance of speed and accuracy
- Process videos in chunks for memory efficiency

## Advanced Usage

### Custom Training Parameters
```python
# Modify training parameters in train_model.py
epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training
learning_rate = 0.001  # Initial learning rate
```

### Model Comparison
Train multiple models and compare their performance:
```bash
# Train different architectures
python train_model.py  # Choose option 1 (CNN)
python train_model.py  # Choose option 2 (EfficientNet)
python train_model.py  # Choose option 3 (ResNet)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- DFDC dataset creators
- TensorFlow and Keras teams
- OpenCV community
- Gradio developers

## Future Enhancements

- [ ] Support for more video formats
- [ ] Real-time webcam detection
- [ ] Mobile app development
- [ ] Advanced preprocessing techniques
- [ ] Ensemble model methods
- [ ] API endpoint creation
