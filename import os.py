import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import json
from pathlib import Path

def check_kaggle_credentials():
    """Check if Kaggle credentials are properly set up"""
    kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_path.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Download kaggle.json")
        print(f"5. Place it in: {kaggle_path}")
        print("\nAlternatively, you can manually create the file with your credentials:")
        print('{"username":"your_username","key":"your_api_key"}')
        return False
    
    try:
        with open(kaggle_path, 'r') as f:
            creds = json.load(f)
        if 'username' in creds and 'key' in creds:
            print("✅ Kaggle credentials found!")
            return True
        else:
            print("❌ Invalid kaggle.json format!")
            return False
    except Exception as e:
        print(f"❌ Error reading kaggle.json: {e}")
        return False

def download_dfdc_preview():
    """Download DFDC Preview dataset from Kaggle"""
    print("=== DFDC Preview Dataset Download ===")
    
    # Check credentials first
    if not check_kaggle_credentials():
        return False
    
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        print("✅ Kaggle API authenticated successfully!")
        
        # Dataset information
        dataset = 'awsaf49/dfdc-preview'
        download_path = 'dfdc_preview'
        
        # Create download directory
        os.makedirs(download_path, exist_ok=True)
        print(f"📁 Download directory: {os.path.abspath(download_path)}")
        
        # Check if dataset already exists
        if os.path.exists(os.path.join(download_path, 'metadata.json')):
            print("⚠️  Dataset already exists!")
            choice = input("Do you want to re-download? (y/N): ").strip().lower()
            if choice != 'y':
                print("Using existing dataset.")
                return True
        
        # Download dataset
        print("📥 Downloading dataset... This may take a while.")
        print("💡 The DFDC Preview dataset is approximately 5.2 GB")
        
        api.dataset_download_files(dataset, path=download_path, unzip=True)
        print("✅ Download complete!")
        
        # Verify download
        metadata_path = os.path.join(download_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"📊 Dataset contains {len(metadata)} videos")
            
            # Count real vs fake
            real_count = sum(1 for v in metadata.values() if v['label'] == 'REAL')
            fake_count = len(metadata) - real_count
            print(f"   - Real videos: {real_count}")
            print(f"   - Fake videos: {fake_count}")
        
        # Extract any remaining zip files
        print("🔍 Checking for zip files to extract...")
        zip_count = 0
        for file in os.listdir(download_path):
            if file.endswith('.zip'):
                zip_path = os.path.join(download_path, file)
                print(f"📦 Extracting {file}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(download_path)
                os.remove(zip_path)  # Remove zip after extraction
                zip_count += 1
        
        if zip_count > 0:
            print(f"✅ Extracted {zip_count} zip files")
        else:
            print("✅ No additional zip files to extract")
        
        print("\n🎉 Dataset ready for use!")
        print(f"📂 Dataset location: {os.path.abspath(download_path)}")
        print("\nNext steps:")
        print("1. Run: python train_model.py")
        print("2. Or run: python train_model.py test (to verify setup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Verify Kaggle credentials")
        print("3. Ensure you have accepted the dataset's terms")
        print("4. Check available disk space (need ~6GB)")
        return False

def get_dataset_info():
    """Get information about the downloaded dataset"""
    download_path = 'dfdc_preview'
    
    if not os.path.exists(download_path):
        print("❌ Dataset not found. Please download it first.")
        return
    
    print("=== Dataset Information ===")
    
    # Check metadata
    metadata_path = os.path.join(download_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"📊 Total videos: {len(metadata)}")
        
        # Count by label
        labels = {}
        for video_info in metadata.values():
            label = video_info['label']
            labels[label] = labels.get(label, 0) + 1
        
        for label, count in labels.items():
            print(f"   - {label}: {count}")
    
    # Count video files
    video_files = [f for f in os.listdir(download_path) if f.endswith('.mp4')]
    print(f"📹 Video files found: {len(video_files)}")
    
    # Calculate total size
    total_size = 0
    for file in os.listdir(download_path):
        file_path = os.path.join(download_path, file)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    
    size_gb = total_size / (1024**3)
    print(f"💾 Total size: {size_gb:.2f} GB")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'info':
        get_dataset_info()
    else:
        success = download_dfdc_preview()
        if success:
            get_dataset_info()