"""
Quick test script to verify the complete system works
"""

import cv2
import numpy as np
import requests
import json
from pathlib import Path

def create_test_image():
    """Create a simple test image"""
    # Create a simple face-like image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (220, 215, 200)  # Background
    
    # Face oval
    cv2.ellipse(img, (100, 100), (80, 90), 0, 0, 360, (200, 190, 180), -1)
    
    # Eyes
    cv2.circle(img, (75, 80), 8, (50, 50, 50), -1)
    cv2.circle(img, (125, 80), 8, (50, 50, 50), -1)
    
    # Nose
    pts = np.array([[100, 90], [95, 110], [105, 110]], np.int32)
    cv2.fillPoly(img, [pts], (180, 170, 160))
    
    # Mouth
    cv2.ellipse(img, (100, 130), (15, 8), 0, 0, 180, (150, 100, 100), -1)
    
    # Save test image
    test_image_path = "test_image.jpg"
    cv2.imwrite(test_image_path, img)
    return test_image_path

def test_api_directly():
    """Test the API directly"""
    print("🧪 Testing API directly...")
    
    # Create test image
    test_image = create_test_image()
    
    try:
        # Test status endpoint
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            print("✅ API Status:", response.json()['status'])
        else:
            print("❌ API not responding")
            return False
        
        # Test prediction endpoint
        with open(test_image, 'rb') as f:
            files = {'image': f}
            response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Result: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            print(f"   Processing time: {result['processingTime']}s")
            return True
        else:
            print("❌ Prediction failed:", response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server")
        print("💡 Start the server with: python api_server.py")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Clean up test image
        try:
            Path(test_image).unlink()
        except:
            pass

def test_simple_detector():
    """Test the simple detector directly"""
    print("\n🔬 Testing Simple Detector directly...")
    
    try:
        from simple_detector import SimpleDeepfakeDetector
        
        detector = SimpleDeepfakeDetector()
        if detector.load_model('simple_deepfake_model.pkl'):
            # Create test image
            test_image = create_test_image()
            
            # Make prediction
            label, confidence = detector.predict_image(test_image)
            print("✅ Simple detector working!")
            print(f"   Result: {label}")
            print(f"   Confidence: {confidence:.2f}")
            
            # Clean up
            Path(test_image).unlink()
            return True
        else:
            print("❌ Could not load simple model")
            return False
            
    except Exception as e:
        print(f"❌ Simple detector test failed: {e}")
        return False

def main():
    print("🚀 Deepfake Detection System Test")
    print("=" * 40)
    
    # Test simple detector
    detector_works = test_simple_detector()
    
    # Test API
    api_works = test_api_directly()
    
    print("\n📋 Test Summary:")
    print(f"   Simple Detector: {'✅ Working' if detector_works else '❌ Failed'}")
    print(f"   API Server: {'✅ Working' if api_works else '❌ Failed'}")
    
    if detector_works and api_works:
        print("\n🎉 All systems working! Your frontend is ready to use.")
        print("🌐 Open: http://localhost:8000/frontend")
    elif detector_works and not api_works:
        print("\n⚠️  Detector works but API server is not running.")
        print("💡 Start with: python api_server.py")
    else:
        print("\n❌ System needs setup. Run: python create_dummy_data.py")

if __name__ == "__main__":
    main()
