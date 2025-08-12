#!/usr/bin/env python3
"""
Test script to verify our environment setup
"""

print("Testing environment setup...")

try:
    import flask
    print(f"✅ Flask imported successfully - version: {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask import failed: {e}")

try:
    import cv2
    print(f"✅ OpenCV imported successfully - version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print(f"✅ NumPy imported successfully - version: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn imported successfully - version: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    from flask_cors import CORS
    print("✅ Flask-CORS imported successfully")
except ImportError as e:
    print(f"❌ Flask-CORS import failed: {e}")

print("\nEnvironment test completed!")
