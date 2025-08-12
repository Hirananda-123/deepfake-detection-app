import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_path='dfdc_preview', image_size=(224, 224)):
        self.data_path = data_path
        self.image_size = image_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_faces_from_video(self, video_path, max_frames=30):
        """Extract faces from video frames"""
        faces = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.read()[0] and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in detected_faces:
                # Extract face region
                face = frame[y:y+h, x:x+w]
                # Resize to standard size
                face_resized = cv2.resize(face, self.image_size)
                faces.append(face_resized)
                
            frame_count += 1
            
        cap.release()
        return faces
    
    def load_metadata(self):
        """Load metadata from JSON file"""
        metadata_path = os.path.join(self.data_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def prepare_dataset(self):
        """Prepare the dataset for training"""
        metadata = self.load_metadata()
        
        X = []  # Images
        y = []  # Labels (0 for real, 1 for fake)
        
        # Process videos in the dataset
        for filename in tqdm(os.listdir(self.data_path), desc="Processing videos"):
            if filename.endswith('.mp4'):
                video_path = os.path.join(self.data_path, filename)
                
                # Get label from metadata
                if filename in metadata:
                    label = 1 if metadata[filename]['label'] == 'FAKE' else 0
                else:
                    # If no metadata, skip this file
                    continue
                
                # Extract faces from video
                faces = self.extract_faces_from_video(video_path)
                
                for face in faces:
                    # Normalize pixel values
                    face_normalized = face.astype('float32') / 255.0
                    X.append(face_normalized)
                    y.append(label)
        
        return np.array(X), np.array(y)
    
    def create_data_generators(self, X_train, X_val, y_train, y_val, batch_size=32):
        """Create data generators for training"""
        # Data augmentation for training set
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation set
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator
    
    def visualize_data_distribution(self, y):
        """Visualize the distribution of real vs fake samples"""
        unique, counts = np.unique(y, return_counts=True)
        labels = ['Real', 'Fake']
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, counts)
        plt.title('Distribution of Real vs Fake Samples')
        plt.ylabel('Number of Samples')
        plt.show()
        
        print(f"Real samples: {counts[0]}")
        print(f"Fake samples: {counts[1]}")
        print(f"Total samples: {len(y)}")

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Check if data directory exists
    if os.path.exists(preprocessor.data_path):
        print("Preparing dataset...")
        X, y = preprocessor.prepare_dataset()
        
        if len(X) > 0:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            
            # Visualize data distribution
            preprocessor.visualize_data_distribution(y)
            
            # Save processed data
            np.save('X_train.npy', X_train)
            np.save('X_val.npy', X_val)
            np.save('y_train.npy', y_train)
            np.save('y_val.npy', y_val)
            print("Processed data saved!")
        else:
            print("No data found. Please download the dataset first.")
    else:
        print("Data directory not found. Please run the download script first.")
