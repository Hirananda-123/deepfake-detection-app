"""
Create dummy training data for demonstration purposes
This generates simple synthetic images to train the basic model
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_dummy_face_images():
    """Create dummy face-like images for training"""
    print("Creating dummy training data...")
    
    # Ensure directories exist
    real_dir = Path("sample_data/real")
    fake_dir = Path("sample_data/fake")
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    
    # Create some "real" face-like images
    print("Generating 'real' images...")
    for i in range(15):
        # Create a simple face-like pattern
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Background
        img[:] = (220, 215, 200)  # Skin-like color
        
        # Face oval
        center = (100, 100)
        axes = (80, 90)
        cv2.ellipse(img, center, axes, 0, 0, 360, (200, 190, 180), -1)
        
        # Eyes
        cv2.circle(img, (75, 80), 8, (50, 50, 50), -1)  # Left eye
        cv2.circle(img, (125, 80), 8, (50, 50, 50), -1)  # Right eye
        
        # Nose
        pts = np.array([[100, 90], [95, 110], [105, 110]], np.int32)
        cv2.fillPoly(img, [pts], (180, 170, 160))
        
        # Mouth
        cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (150, 100, 100), -1)
        
        # Add some random variation
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save image
        filename = f"real_face_{i+1:02d}.jpg"
        cv2.imwrite(str(real_dir / filename), img)
    
    # Create some "fake" face-like images with distortions
    print("Generating 'fake' images...")
    for i in range(15):
        # Create a distorted face-like pattern
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        # Background with different color
        img[:] = (200, 220, 210)  # Slightly different skin tone
        
        # Asymmetric face
        center = (100, 100)
        axes = (85, 85)  # More circular (less natural)
        cv2.ellipse(img, center, axes, 0, 0, 360, (180, 200, 190), -1)
        
        # Asymmetric eyes
        cv2.circle(img, (70, 75), 10, (40, 40, 40), -1)  # Left eye bigger
        cv2.circle(img, (130, 85), 6, (40, 40, 40), -1)   # Right eye smaller and off
        
        # Distorted nose
        pts = np.array([[105, 85], [90, 115], [110, 115]], np.int32)
        cv2.fillPoly(img, [pts], (160, 180, 170))
        
        # Odd mouth
        cv2.ellipse(img, (105, 135), (20, 5), 15, 0, 180, (120, 80, 80), -1)
        
        # Add more noise and artifacts
        noise = np.random.normal(0, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add some "glitch" artifacts
        if np.random.random() > 0.5:
            # Random rectangles to simulate compression artifacts
            x1, y1 = np.random.randint(0, 150, 2)
            x2, y2 = x1 + np.random.randint(10, 50), y1 + np.random.randint(10, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # Save image
        filename = f"fake_face_{i+1:02d}.jpg"
        cv2.imwrite(str(fake_dir / filename), img)
    
    print("✅ Dummy training data created!")
    print(f"   - Real images: {len(list(real_dir.glob('*.jpg')))}")
    print(f"   - Fake images: {len(list(fake_dir.glob('*.jpg')))}")
    print("\nNote: These are synthetic images for demonstration only.")
    print("For real-world use, please use actual training data.")

def train_dummy_model():
    """Train the model with dummy data"""
    print("\n" + "="*50)
    print("Training model with dummy data...")
    
    from simple_detector import SimpleDeepfakeDetector
    
    detector = SimpleDeepfakeDetector()
    detector.train_simple_model('sample_data')
    
    if detector.trained:
        print("✅ Model trained successfully!")
        print("You can now test the frontend with this basic model.")
    else:
        print("❌ Model training failed.")

if __name__ == "__main__":
    print("🎭 Dummy Data Generator for Deepfake Detection Demo")
    print("="*55)
    print()
    
    print("Creating dummy training data automatically...")
    create_dummy_face_images()
    
    print("\nTraining model with dummy data...")
    train_dummy_model()
