"""
Simple Deepfake Detection Demo using OpenCV and traditional ML approaches
This version doesn't require TensorFlow and can run immediately
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

class SimpleDeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.trained = False
    
    def extract_face_features(self, image_path):
        """Extract simple features from face images"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # If no face detected, use whole image
                face_region = gray
            else:
                # Use largest detected face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_region = gray[y:y+h, x:x+w]
            
            # Resize to standard size
            face_resized = cv2.resize(face_region, (64, 64))
            
            # Extract features
            features = []
            
            # 1. Basic statistical features
            features.append(np.mean(face_resized))
            features.append(np.std(face_resized))
            features.append(np.median(face_resized))
            features.append(np.min(face_resized))
            features.append(np.max(face_resized))
            
            # 2. Texture features using LBP (simplified)
            lbp_features = self.calculate_lbp_features(face_resized)
            features.extend(lbp_features)
            
            # 3. Edge features
            edges = cv2.Canny(face_resized, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 4. Gradient features
            grad_x = cv2.Sobel(face_resized, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(face_resized, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features.append(np.mean(gradient_magnitude))
            features.append(np.std(gradient_magnitude))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def calculate_lbp_features(self, image):
        """Calculate simplified Local Binary Pattern features"""
        h, w = image.shape
        lbp_features = []
        
        # Sample points around the image
        for i in range(1, h-1, 8):
            for j in range(1, w-1, 8):
                center = image[i, j]
                binary_string = ""
                
                # 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                
                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'
                
                lbp_features.append(int(binary_string, 2))
        
        # Return histogram of LBP values
        hist, _ = np.histogram(lbp_features, bins=32, range=(0, 256))
        return hist.tolist()
    
    def train_simple_model(self, data_path='sample_data'):
        """Train a simple model using sample data"""
        print("Training simple deepfake detector...")
        
        # Create sample data if it doesn't exist
        if not os.path.exists(data_path):
            self.create_sample_data(data_path)
        
        # Load features and labels
        X = []
        y = []
        
        # Process real images
        real_path = os.path.join(data_path, 'real')
        if os.path.exists(real_path):
            for img_file in os.listdir(real_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(real_path, img_file)
                    features = self.extract_face_features(img_path)
                    if features is not None:
                        X.append(features)
                        y.append(0)  # Real = 0
        
        # Process fake images
        fake_path = os.path.join(data_path, 'fake')
        if os.path.exists(fake_path):
            for img_file in os.listdir(fake_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(fake_path, img_file)
                    features = self.extract_face_features(img_path)
                    if features is not None:
                        X.append(features)
                        y.append(1)  # Fake = 1
        
        if len(X) == 0:
            print("No training data found. Please add images to sample_data/real and sample_data/fake folders.")
            return
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Training with {len(X)} samples...")
        print(f"Real samples: {np.sum(y == 0)}")
        print(f"Fake samples: {np.sum(y == 1)}")
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        self.trained = True
        
        # Save model
        with open('simple_deepfake_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved as 'simple_deepfake_model.pkl'")
    
    def predict_image(self, image_path):
        """Predict if an image is real or fake"""
        if not self.trained and self.model is None:
            print("Model not trained. Please train the model first.")
            return None, 0.0
        
        features = self.extract_face_features(image_path)
        if features is None:
            return "Error", 0.0
        
        prediction = self.model.predict([features])[0]
        confidence = max(self.model.predict_proba([features])[0])
        
        label = "FAKE" if prediction == 1 else "REAL"
        return label, confidence
    
    def load_model(self, model_path='simple_deepfake_model.pkl'):
        """Load a trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.trained = True
            print(f"Model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_path} not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def create_sample_data(self, data_path):
        """Create sample directory structure"""
        os.makedirs(os.path.join(data_path, 'real'), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'fake'), exist_ok=True)
        
        readme_content = """
# Sample Data Directory

To train the simple deepfake detector, add images to these folders:

## real/
Add real/authentic face images here
- Supported formats: .jpg, .jpeg, .png
- Images should contain clear faces
- Add at least 10-20 images for better training

## fake/
Add fake/manipulated face images here
- Supported formats: .jpg, .jpeg, .png  
- Images should contain clear faces
- Add at least 10-20 images for better training

## Tips:
- More diverse data = better model
- Images should be reasonably sized (not too small)
- Clear, well-lit faces work best
"""
        
        with open(os.path.join(data_path, 'README.md'), 'w') as f:
            f.write(readme_content)
        
        print(f"Sample data directory created at {data_path}")
        print("Please add real and fake face images to train the model.")

def demo_application():
    """Demo application for simple deepfake detection"""
    print("=== Simple Deepfake Detection Demo ===")
    print("This demo uses traditional computer vision and machine learning.")
    print("No deep learning frameworks required!\n")
    
    detector = SimpleDeepfakeDetector()
    
    # Try to load existing model
    if detector.load_model():
        print("Existing model loaded successfully!")
    else:
        print("No existing model found. Training new model...")
        detector.train_simple_model()
        
        if not detector.trained:
            print("\nTo use this detector:")
            print("1. Add real face images to: sample_data/real/")
            print("2. Add fake face images to: sample_data/fake/")
            print("3. Run this script again to train the model")
            return
    
    # Demo prediction
    while True:
        print("\n--- Deepfake Detection Demo ---")
        print("1. Predict single image")
        print("2. Retrain model")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                label, confidence = detector.predict_image(image_path)
                print(f"Prediction: {label}")
                print(f"Confidence: {confidence:.2f}")
                
                # Visualize result
                try:
                    img = cv2.imread(image_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img_rgb)
                    plt.title(f"Prediction: {label} (Confidence: {confidence:.2f})")
                    plt.axis('off')
                    plt.show()
                except Exception as e:
                    print(f"Could not display image: {e}")
            else:
                print("Image file not found!")
        
        elif choice == '2':
            detector.train_simple_model()
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    demo_application()
