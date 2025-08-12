"""
Advanced Deepfake Detection using multiple detection techniques
Combines traditional CV features with modern approaches for better AI image detection
"""

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
import os
import pickle
from scipy import stats, ndimage
from skimage import feature, filters, measure
import warnings
warnings.filterwarnings('ignore')

class AdvancedDeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = None
        self.scaler = None
        self.trained = False
        
    def extract_advanced_features(self, image_path):
        """Extract comprehensive features designed to detect AI-generated images"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Detect face region
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                # Use largest detected face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                face_gray = gray[y:y+h, x:x+w]
                face_color = img[y:y+h, x:x+w]
            else:
                # Use center region if no face detected
                h, w = gray.shape
                face_gray = gray[h//4:3*h//4, w//4:3*w//4]
                face_color = img[h//4:3*h//4, w//4:3*w//4]
            
            # Resize for consistent analysis
            face_gray = cv2.resize(face_gray, (128, 128))
            face_color = cv2.resize(face_color, (128, 128))
            
            features = []
            
            # 1. STATISTICAL FEATURES (AI images often have different statistical properties)
            features.extend(self._statistical_features(face_gray))
            
            # 2. TEXTURE ANALYSIS (AI generated textures have subtle differences)
            features.extend(self._texture_features(face_gray))
            
            # 3. FREQUENCY DOMAIN FEATURES (AI artifacts visible in frequency domain)
            features.extend(self._frequency_features(face_gray))
            
            # 4. COLOR CONSISTENCY FEATURES (AI may have color inconsistencies)
            features.extend(self._color_features(face_color))
            
            # 5. EDGE AND GRADIENT FEATURES (AI edges may be too perfect or inconsistent)
            features.extend(self._edge_features(face_gray))
            
            # 6. COMPRESSION ARTIFACTS (Real photos have different compression patterns)
            features.extend(self._compression_features(face_gray))
            
            # 7. SYMMETRY FEATURES (AI faces may be too symmetric)
            features.extend(self._symmetry_features(face_gray))
            
            # 8. NOISE ANALYSIS (AI images have different noise patterns)
            features.extend(self._noise_features(face_gray))
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _statistical_features(self, img):
        """Extract statistical features"""
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(img), np.std(img), np.var(img),
            np.min(img), np.max(img), np.median(img),
            stats.skew(img.flatten()), stats.kurtosis(img.flatten())
        ])
        
        # Histogram features
        hist = cv2.calcHist([img], [0], None, [64], [0, 256])
        features.extend(hist.flatten()[:20])  # First 20 bins
        
        # Percentiles
        percentiles = [10, 25, 75, 90]
        for p in percentiles:
            features.append(np.percentile(img, p))
        
        return features
    
    def _texture_features(self, img):
        """Extract texture features using multiple methods"""
        features = []
        
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(img, 24, 8, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 25))
        features.extend(lbp_hist)
        
        # Gray Level Co-occurrence Matrix features
        try:
            glcm = feature.graycomatrix(img.astype(np.uint8), [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            features.extend([
                feature.graycoprops(glcm, 'contrast')[0, 0],
                feature.graycoprops(glcm, 'dissimilarity')[0, 0],
                feature.graycoprops(glcm, 'homogeneity')[0, 0],
                feature.graycoprops(glcm, 'energy')[0, 0]
            ])
        except:
            features.extend([0, 0, 0, 0])
        
        return features
    
    def _frequency_features(self, img):
        """Extract frequency domain features"""
        features = []
        
        # FFT analysis
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # High frequency energy (AI images may have less high freq noise)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h//2, w//2
        high_freq_region = magnitude_spectrum.copy()
        high_freq_region[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8] = 0
        
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.mean(high_freq_region),
            np.std(high_freq_region)
        ])
        
        # DCT features
        dct = cv2.dct(np.float32(img))
        features.extend([
            np.mean(dct), np.std(dct),
            np.mean(dct[:32, :32]), np.std(dct[:32, :32])  # Low frequency
        ])
        
        return features
    
    def _color_features(self, img):
        """Extract color consistency features"""
        features = []
        
        if len(img.shape) == 3:
            # Color channel analysis
            b, g, r = cv2.split(img)
            
            # Color consistency across channels
            features.extend([
                np.corrcoef(r.flatten(), g.flatten())[0, 1],
                np.corrcoef(r.flatten(), b.flatten())[0, 1],
                np.corrcoef(g.flatten(), b.flatten())[0, 1]
            ])
            
            # Color variance
            features.extend([np.std(r), np.std(g), np.std(b)])
            
            # Color ratios
            features.extend([
                np.mean(r) / (np.mean(g) + 1e-6),
                np.mean(g) / (np.mean(b) + 1e-6),
                np.mean(r) / (np.mean(b) + 1e-6)
            ])
        else:
            features.extend([0] * 9)
        
        return features
    
    def _edge_features(self, img):
        """Extract edge and gradient features"""
        features = []
        
        # Canny edges
        edges = cv2.Canny(img, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # Sobel gradients
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        features.extend([
            np.mean(gradient_magnitude), np.std(gradient_magnitude),
            np.mean(np.abs(grad_x)), np.std(np.abs(grad_x)),
            np.mean(np.abs(grad_y)), np.std(np.abs(grad_y))
        ])
        
        # Edge orientation consistency
        direction_hist, _ = np.histogram(gradient_direction, bins=8, range=(-np.pi, np.pi))
        features.extend(direction_hist / np.sum(direction_hist))
        
        return features
    
    def _compression_features(self, img):
        """Extract compression artifact features"""
        features = []
        
        # Block-based analysis (JPEG artifacts)
        block_size = 8
        h, w = img.shape
        
        block_variances = []
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block = img[i:i+block_size, j:j+block_size]
                block_variances.append(np.var(block))
        
        if block_variances:
            features.extend([
                np.mean(block_variances),
                np.std(block_variances),
                np.max(block_variances) - np.min(block_variances)
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _symmetry_features(self, img):
        """Extract facial symmetry features"""
        features = []
        
        h, w = img.shape
        left_half = img[:, :w//2]
        right_half = np.fliplr(img[:, w//2:])
        
        # Resize to same size if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Symmetry correlation
        if left_half.shape == right_half.shape:
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            features.append(correlation if not np.isnan(correlation) else 0)
            
            # Symmetry difference
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            features.extend([np.mean(diff), np.std(diff)])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _noise_features(self, img):
        """Extract noise pattern features"""
        features = []
        
        # Gaussian noise estimation
        noise = img - cv2.GaussianBlur(img, (5, 5), 1.0)
        features.extend([np.std(noise), np.mean(np.abs(noise))])
        
        # High-pass filter response
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        high_pass = cv2.filter2D(img, -1, kernel)
        features.extend([np.std(high_pass), np.mean(np.abs(high_pass))])
        
        return features
    
    def train_advanced_model(self, data_path='sample_data'):
        """Train an ensemble model with advanced features"""
        print("Training advanced deepfake detector...")
        
        # Load features and labels
        X, y = self._load_training_data(data_path)
        
        if len(X) == 0:
            print("No training data found. Please add images to sample_data/real and sample_data/fake folders.")
            return False
        
        print(f"Training with {len(X)} samples ({np.sum(y == 0)} real, {np.sum(y == 1)} fake)...")
        
        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        # Create ensemble model with multiple algorithms
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=8, random_state=42)
        svm = SVC(probability=True, kernel='rbf', random_state=42)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('svm', svm)],
            voting='soft'
        )
        
        # Create pipeline with scaling
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('ensemble', ensemble)
        ])
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Advanced model trained! Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        # Feature importance (from Random Forest)
        rf_model = self.model.named_steps['ensemble'].named_estimators_['rf']
        importances = rf_model.feature_importances_
        top_features = np.argsort(importances)[-10:]
        print("\nTop 10 Most Important Features:")
        feature_names = self._get_feature_names()
        for i, idx in enumerate(reversed(top_features)):
            print(f"{i+1:2d}. Feature {idx:3d}: {importances[idx]:.4f}")
        
        self.trained = True
        
        # Save model
        with open('advanced_deepfake_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("\nModel saved as 'advanced_deepfake_model.pkl'")
        
        return True
    
    def _load_training_data(self, data_path):
        """Load and extract features from training data"""
        X, y = [], []
        
        # Process real images
        real_path = os.path.join(data_path, 'real')
        if os.path.exists(real_path):
            print("Processing real images...")
            for i, img_file in enumerate(os.listdir(real_path)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(real_path, img_file)
                    features = self.extract_advanced_features(img_path)
                    if features is not None:
                        X.append(features)
                        y.append(0)  # Real = 0
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1} real images...")
        
        # Process fake images
        fake_path = os.path.join(data_path, 'fake')
        if os.path.exists(fake_path):
            print("Processing fake images...")
            for i, img_file in enumerate(os.listdir(fake_path)):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(fake_path, img_file)
                    features = self.extract_advanced_features(img_path)
                    if features is not None:
                        X.append(features)
                        y.append(1)  # Fake = 1
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1} fake images...")
        
        return np.array(X), np.array(y)
    
    def _get_feature_names(self):
        """Get names of extracted features"""
        names = []
        names.extend(['mean', 'std', 'var', 'min', 'max', 'median', 'skew', 'kurtosis'])
        names.extend([f'hist_{i}' for i in range(20)])
        names.extend(['p10', 'p25', 'p75', 'p90'])
        names.extend([f'lbp_{i}' for i in range(26)])
        names.extend(['glcm_contrast', 'glcm_dissim', 'glcm_homogen', 'glcm_energy'])
        names.extend(['fft_mean', 'fft_std', 'fft_hf_mean', 'fft_hf_std'])
        names.extend(['dct_mean', 'dct_std', 'dct_lf_mean', 'dct_lf_std'])
        names.extend(['color_corr_rg', 'color_corr_rb', 'color_corr_gb'])
        names.extend(['color_std_r', 'color_std_g', 'color_std_b'])
        names.extend(['color_ratio_rg', 'color_ratio_gb', 'color_ratio_rb'])
        names.extend(['edge_density'])
        names.extend(['grad_mean', 'grad_std', 'gradx_mean', 'gradx_std', 'grady_mean', 'grady_std'])
        names.extend([f'orient_{i}' for i in range(8)])
        names.extend(['block_var_mean', 'block_var_std', 'block_var_range'])
        names.extend(['symmetry_corr', 'symmetry_diff_mean', 'symmetry_diff_std'])
        names.extend(['noise_std', 'noise_mean', 'highpass_std', 'highpass_mean'])
        return names
    
    def predict_image(self, image_path):
        """Predict if an image is real or fake using advanced features"""
        if not self.trained and self.model is None:
            return "Error: Model not trained", 0.0
        
        features = self.extract_advanced_features(image_path)
        if features is None:
            return "Error: Could not process image", 0.0
        
        try:
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            label = "FAKE" if prediction == 1 else "REAL"
            return label, confidence
        except Exception as e:
            return f"Error: {str(e)}", 0.0
    
    def load_model(self, model_path='advanced_deepfake_model.pkl'):
        """Load a trained advanced model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.trained = True
            print(f"Advanced model loaded from {model_path}")
            return True
        except FileNotFoundError:
            print(f"Model file {model_path} not found.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def demo_advanced_detector():
    """Demo the advanced detector"""
    print("=== Advanced Deepfake Detection ===")
    print("Using comprehensive feature analysis for better AI detection\n")
    
    detector = AdvancedDeepfakeDetector()
    
    # Try to load existing model
    if detector.load_model():
        print("✅ Advanced model loaded successfully!")
    else:
        print("No existing advanced model found. Training new model...")
        if detector.train_advanced_model():
            print("✅ Advanced model trained successfully!")
        else:
            print("❌ Training failed. Please add training data.")
            return
    
    # Demo prediction
    while True:
        print("\n--- Advanced Deepfake Detection ---")
        print("1. Analyze image")
        print("2. Retrain model")
        print("3. Exit")
        
        choice = input("Choose option (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if os.path.exists(image_path):
                print("Analyzing image with advanced features...")
                label, confidence = detector.predict_image(image_path)
                print(f"\n🔍 Analysis Result:")
                print(f"   Prediction: {label}")
                print(f"   Confidence: {confidence:.3f}")
                
                if "FAKE" in label:
                    print("   ⚠️  This image may be AI-generated or manipulated")
                else:
                    print("   ✅ This image appears to be authentic")
            else:
                print("❌ Image file not found!")
        
        elif choice == '2':
            detector.train_advanced_model()
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    demo_advanced_detector()
