"""
Complete Deepfake Detection Application Demo
This script demonstrates all the functionality of our deepfake detection system
"""

import os
import sys
from pathlib import Path

def print_header():
    print("=" * 60)
    print("     🔍 DEEPFAKE DETECTION SYSTEM DEMO")
    print("=" * 60)
    print()

def print_menu():
    print("Available Options:")
    print("1. 📥 Download DFDC Dataset")
    print("2. 🧠 Train Deep Learning Model (Requires TensorFlow)")
    print("3. 🎯 Simple Traditional ML Demo (Works immediately)")
    print("4. 🌐 Launch Gradio Web Application")
    print("5. 🖥️  Launch HTML Frontend with API Server")
    print("6. 📊 Check Dataset Information") 
    print("7. ❓ Help & Setup Instructions")
    print("8. 🚪 Exit")
    print()

def download_dataset():
    """Download the DFDC dataset"""
    print("Downloading DFDC dataset...")
    try:
        exec(open('import os.py').read())
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have:")
        print("1. Kaggle account and API credentials")
        print("2. Internet connection")
        print("3. Sufficient disk space (~6GB)")

def train_deep_model():
    """Train the deep learning model"""
    print("Training deep learning model...")
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        os.system('python train_model.py')
    except ImportError:
        print("❌ TensorFlow not available!")
        print("Please install TensorFlow to use deep learning models:")
        print("pip install tensorflow")
        print()
        print("Alternative: Use the Simple Traditional ML Demo (Option 3)")

def run_simple_demo():
    """Run the simple traditional ML demo"""
    print("Running simple traditional ML demo...")
    os.system('python simple_detector.py')

def launch_web_app():
    """Launch the Gradio web application"""
    print("Launching Gradio web application...")
    
    # Check if models exist
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    if not model_files and not pkl_files:
        print("❌ No trained models found!")
        print("Please train a model first using option 2 or 3")
        return
    
    try:
        import gradio
        os.system('python web_app.py')
    except ImportError:
        print("❌ Gradio not available!")
        print("Installing Gradio...")
        os.system('pip install gradio')
        os.system('python web_app.py')

def launch_html_frontend():
    """Launch the HTML frontend with API server"""
    print("Launching HTML Frontend with API Server...")
    
    # Check if models exist
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    if not model_files and not pkl_files:
        print("❌ No trained models found!")
        print("Please train a model first using option 2 or 3")
        return
    
    try:
        print("Starting API server...")
        print("🌐 Frontend will be available at: http://localhost:8000/frontend")
        print("📡 API will be available at: http://localhost:8000")
        print("⏹️  Press Ctrl+C to stop the server")
        os.system('python api_server.py')
    except Exception as e:
        print(f"❌ Error launching frontend: {e}")
        print("Make sure Flask is installed: pip install flask flask-cors")

def check_dataset_info():
    """Check dataset information"""
    print("Checking dataset information...")
    os.system('python "import os.py" info')

def show_help():
    """Show help and setup instructions"""
    print("🆘 HELP & SETUP INSTRUCTIONS")
    print("=" * 40)
    print()
    
    print("📋 SYSTEM REQUIREMENTS:")
    print("- Python 3.8 or higher")
    print("- OpenCV (opencv-python)")
    print("- NumPy, Matplotlib, Scikit-learn")
    print("- TensorFlow (optional, for deep learning)")
    print("- Gradio (for web interface)")
    print()
    
    print("🚀 QUICK START GUIDE:")
    print("1. First time setup:")
    print("   - Run option 3 (Simple Demo) - works immediately")
    print("   - Add sample images to train a basic model")
    print()
    
    print("2. For advanced deep learning:")
    print("   - Download dataset (option 1)")
    print("   - Train deep model (option 2)")
    print("   - Launch web app (option 4)")
    print()
    
    print("📁 FILE STRUCTURE:")
    files = [
        "import os.py - Dataset download script",
        "simple_detector.py - Traditional ML approach",
        "deepfake_detector.py - Deep learning models",
        "train_model.py - Training pipeline",
        "web_app.py - Web interface",
        "predictor.py - Prediction utilities",
        "data_preprocessing.py - Data processing"
    ]
    
    for file in files:
        if os.path.exists(file.split(' - ')[0]):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")
    print()
    
    print("🔧 TROUBLESHOOTING:")
    print("- TensorFlow issues: Use Simple Demo (option 3)")
    print("- Kaggle API issues: Check credentials in ~/.kaggle/")
    print("- No training data: Add images to sample_data folder")
    print("- Web app issues: Check if Gradio is installed")
    print()
    
    print("📞 NEED HELP?")
    print("- Check README.md for detailed instructions")
    print("- Ensure all dependencies are installed")
    print("- Start with Simple Demo for immediate results")

def check_system_status():
    """Check system status and dependencies"""
    print("🔍 SYSTEM STATUS CHECK")
    print("-" * 25)
    
    # Check Python packages
    packages = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'scikit-learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'gradio': 'gradio',
        'kaggle': 'kaggle'
    }
    
    for package, module in packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            if package in ['tensorflow', 'gradio']:
                print(f"⚠️  {package} (optional)")
            else:
                print(f"❌ {package} (required)")
    
    # Check dataset
    if os.path.exists('dfdc_preview'):
        print("✅ DFDC dataset downloaded")
    else:
        print("❌ DFDC dataset not found")
    
    # Check models
    model_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.pkl'))]
    if model_files:
        print(f"✅ {len(model_files)} trained model(s) found")
    else:
        print("❌ No trained models found")
    
    # Check sample data
    if os.path.exists('sample_data'):
        real_imgs = len([f for f in os.listdir('sample_data/real') if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists('sample_data/real') else 0
        fake_imgs = len([f for f in os.listdir('sample_data/fake') if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists('sample_data/fake') else 0
        print(f"📊 Sample data: {real_imgs} real, {fake_imgs} fake images")
    else:
        print("❌ No sample data found")
    
    print()

def main():
    """Main application loop"""
    print_header()
    
    # Show system status
    check_system_status()
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-8): ").strip()
        print()
        
        if choice == '1':
            download_dataset()
        elif choice == '2':
            train_deep_model()
        elif choice == '3':
            run_simple_demo()
        elif choice == '4':
            launch_web_app()
        elif choice == '5':
            launch_html_frontend()
        elif choice == '6':
            check_dataset_info()
        elif choice == '7':
            show_help()
        elif choice == '8':
            print("👋 Thank you for using the Deepfake Detection System!")
            print("🔗 Don't forget to check out the README.md for more details")
            break
        else:
            print("❌ Invalid choice! Please enter a number between 1-8.")
        
        input("\nPress Enter to continue...")
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
