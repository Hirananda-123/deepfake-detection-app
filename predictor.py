import cv2
import numpy as np
import os
from deepfake_detector import DeepfakeDetector
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

class DeepfakePredictor:
    def __init__(self, model_path=None):
        self.detector = DeepfakeDetector()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No model loaded. Please train a model first or provide a valid model path.")
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            self.detector.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess an image for prediction"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # If no face detected, resize the whole image
            face = cv2.resize(image, target_size)
        else:
            # Use the largest face detected
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
        
        # Normalize
        face = face.astype('float32') / 255.0
        
        return face
    
    def predict_image(self, image_path):
        """Predict if an image is fake or real"""
        if self.detector.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        label, confidence = self.detector.predict_single_image(processed_image)
        
        return label, confidence
    
    def predict_video(self, video_path, max_frames=30, threshold=0.5):
        """Predict if a video contains deepfakes"""
        if self.detector.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        fake_predictions = 0
        total_predictions = 0
        
        while cap.read()[0] and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face = frame_rgb[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (224, 224))
                face_normalized = face_resized.astype('float32') / 255.0
                
                # Make prediction
                label, confidence = self.detector.predict_single_image(face_normalized)
                
                if label == "FAKE":
                    fake_predictions += 1
                total_predictions += 1
            
            frame_count += 1
        
        cap.release()
        
        if total_predictions == 0:
            return "UNKNOWN", 0.0
        
        fake_ratio = fake_predictions / total_predictions
        
        if fake_ratio > threshold:
            return "FAKE", fake_ratio
        else:
            return "REAL", 1 - fake_ratio
    
    def visualize_prediction(self, image_path):
        """Visualize prediction with face detection"""
        # Read original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Make prediction
        label, confidence = self.predict_image(image_path)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            color = (255, 0, 0) if label == "FAKE" else (0, 255, 0)
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(image_rgb, label_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Display image
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        plt.title(f"Prediction: {label} (Confidence: {confidence:.2f})")
        plt.axis('off')
        plt.show()
        
        return label, confidence
    
    def batch_predict_directory(self, directory_path, output_file="predictions.txt"):
        """Predict all images in a directory"""
        if self.detector.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        results = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                try:
                    label, confidence = self.predict_image(image_path)
                    results.append((filename, label, confidence))
                    print(f"{filename}: {label} (Confidence: {confidence:.2f})")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        # Save results to file
        with open(output_file, 'w') as f:
            f.write("Filename,Prediction,Confidence\n")
            for filename, label, confidence in results:
                f.write(f"{filename},{label},{confidence:.4f}\n")
        
        print(f"Results saved to {output_file}")
        return results

if __name__ == "__main__":
    # Example usage
    predictor = DeepfakePredictor()
    
    # If you have a trained model, load it
    # predictor.load_model("deepfake_detector_best.h5")
    
    print("Deepfake Predictor initialized!")
    print("To use this predictor:")
    print("1. Train a model using deepfake_detector.py")
    print("2. Load the trained model using predictor.load_model('model_path')")
    print("3. Use predictor.predict_image('image_path') for single predictions")
    print("4. Use predictor.predict_video('video_path') for video predictions")
