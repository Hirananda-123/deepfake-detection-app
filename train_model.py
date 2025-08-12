import numpy as np
import os
from data_preprocessing import DataPreprocessor
from deepfake_detector import DeepfakeDetector
from predictor import DeepfakePredictor
import matplotlib.pyplot as plt

def main():
    """Main training pipeline for deepfake detection"""
    
    print("=== Deepfake Detection Training Pipeline ===")
    
    # Step 1: Check if dataset exists
    data_path = 'dfdc_preview'
    if not os.path.exists(data_path):
        print("Dataset not found. Please run the download script first:")
        print("python 'import os.py'")
        return
    
    # Step 2: Data Preprocessing
    print("\n1. Data Preprocessing...")
    preprocessor = DataPreprocessor(data_path=data_path)
    
    # Check if preprocessed data already exists
    if os.path.exists('X_train.npy') and os.path.exists('y_train.npy'):
        print("Loading preprocessed data...")
        X_train = np.load('X_train.npy')
        X_val = np.load('X_val.npy')
        y_train = np.load('y_train.npy')
        y_val = np.load('y_val.npy')
    else:
        print("Preprocessing dataset...")
        X, y = preprocessor.prepare_dataset()
        
        if len(X) == 0:
            print("No data found. Please check your dataset.")
            return
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save preprocessed data
        np.save('X_train.npy', X_train)
        np.save('X_val.npy', X_val)
        np.save('y_train.npy', y_train)
        np.save('y_val.npy', y_val)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Visualize data distribution
    preprocessor.visualize_data_distribution(y_train)
    
    # Step 3: Create data generators
    print("\n2. Creating data generators...")
    train_generator, val_generator = preprocessor.create_data_generators(
        X_train, X_val, y_train, y_val, batch_size=32
    )
    
    # Step 4: Model Selection and Training
    print("\n3. Model Training...")
    detector = DeepfakeDetector()
    
    # Choose model architecture
    print("Select model architecture:")
    print("1. Custom CNN")
    print("2. EfficientNetB0 (Recommended)")
    print("3. ResNet50V2")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        model = detector.create_cnn_model()
        model_name = 'custom_cnn'
    elif choice == '2':
        model = detector.create_efficientnet_model()
        model_name = 'efficientnet'
    elif choice == '3':
        model = detector.create_resnet_model()
        model_name = 'resnet'
    else:
        print("Invalid choice. Using EfficientNetB0 by default.")
        model = detector.create_efficientnet_model()
        model_name = 'efficientnet'
    
    # Compile model
    model = detector.compile_model(model, learning_rate=0.001)
    
    print(f"Training {model_name} model...")
    print("Model Summary:")
    model.summary()
    
    # Train model
    epochs = 30  # Adjust as needed
    history = detector.train_model(
        model, train_generator, val_generator, 
        epochs=epochs, model_name=model_name
    )
    
    # Step 5: Plot training history
    print("\n4. Plotting training history...")
    detector.plot_training_history(history)
    
    # Step 6: Evaluate model
    print("\n5. Model Evaluation...")
    val_loss, val_accuracy, val_precision, val_recall = model.evaluate(
        X_val, y_val, verbose=1
    )
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    
    # Calculate F1 Score
    f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
    print(f"F1 Score: {f1_score:.4f}")
    
    # Step 7: Save final model
    final_model_path = f"{model_name}_final.h5"
    model.save(final_model_path)
    print(f"Final model saved as {final_model_path}")
    
    # Step 8: Demo prediction
    print("\n6. Demo Prediction...")
    predictor = DeepfakePredictor(model_path=f"{model_name}_best.h5")
    
    # If you have test images, you can add them here
    test_image_path = input("Enter path to a test image (or press Enter to skip): ").strip()
    if test_image_path and os.path.exists(test_image_path):
        try:
            label, confidence = predictor.visualize_prediction(test_image_path)
            print(f"Prediction: {label} with confidence {confidence:.4f}")
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    print("\nTraining completed successfully!")
    print(f"Best model saved as: {model_name}_best.h5")
    print(f"Final model saved as: {final_model_path}")

def quick_test():
    """Quick test function to verify setup"""
    print("=== Quick Setup Test ===")
    
    # Test imports
    try:
        import tensorflow as tf
        import cv2
        import numpy as np
        print("✓ All required packages imported successfully")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return
    
    # Test GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("✓ GPU is available")
    else:
        print("⚠ GPU not available, using CPU")
    
    # Test model creation
    try:
        detector = DeepfakeDetector()
        model = detector.create_cnn_model()
        print("✓ Model creation successful")
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return
    
    print("✓ Setup test passed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        quick_test()
    else:
        main()
