import gradio as gr
import numpy as np
import cv2
from predictor import DeepfakePredictor
from simple_detector import SimpleDeepfakeDetector
import os
import tempfile
import pickle

class DeepfakeWebApp:
    def __init__(self, model_path=None, use_simple_model=False):
        self.use_simple_model = use_simple_model
        
        if use_simple_model:
            self.simple_detector = SimpleDeepfakeDetector()
            # Try to load simple model
            if os.path.exists('simple_deepfake_model.pkl'):
                self.simple_detector.load_model('simple_deepfake_model.pkl')
            else:
                print("No simple model found. Please train one first.")
        else:
            self.predictor = DeepfakePredictor(model_path)
        
    def predict_image(self, image):
        """Predict if an uploaded image is fake or real"""
        try:
            if image is None:
                return "Please upload an image", 0.0, None
            
            # Save uploaded image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                
                # Make prediction based on model type
                if self.use_simple_model:
                    if not self.simple_detector.trained:
                        return "Simple model not trained. Please train it first.", 0.0, None
                    label, confidence = self.simple_detector.predict_image(tmp_file.name)
                else:
                    if self.predictor.detector.model is None:
                        return "Deep learning model not loaded. Please load a model first.", 0.0, None
                    label, confidence = self.predictor.predict_image(tmp_file.name)
                
                # Create visualization
                result_image = self.create_result_visualization(tmp_file.name, label, confidence)
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
                return f"Prediction: {label}", confidence, result_image
        except Exception as e:
            return f"Error: {str(e)}", 0.0, None
    
    def create_result_visualization(self, image_path, label, confidence):
        """Create a visualization of the prediction result"""
        try:
            # Read the image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces for visualization
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                color = (255, 0, 0) if label == "FAKE" else (0, 255, 0)  # Red for fake, Green for real
                cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 3)
                
                # Add label text
                label_text = f"{label}: {confidence:.2f}"
                font_scale = max(0.7, min(2.0, w / 200))  # Scale font based on face size
                cv2.putText(img_rgb, label_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            # If no faces detected, add text at top
            if len(faces) == 0:
                color = (255, 0, 0) if label == "FAKE" else (0, 255, 0)
                height, width = img_rgb.shape[:2]
                cv2.putText(img_rgb, f"{label}: {confidence:.2f}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            return img_rgb
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def predict_video(self, video):
        """Predict if an uploaded video contains deepfakes"""
        try:
            if video is None:
                return "Please upload a video", 0.0, None
            
            # Make prediction based on model type
            if self.use_simple_model:
                return "Video analysis not supported with simple model", 0.0, None
            else:
                if self.predictor.detector.model is None:
                    return "Deep learning model not loaded. Please load a model first.", 0.0, None
                label, confidence = self.predictor.predict_video(video)
            
            return f"Prediction: {label}", confidence, None
        except Exception as e:
            return f"Error: {str(e)}", 0.0, None
    
    def create_interface(self):
        """Create Gradio interface"""
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .result-positive {
            color: #10b981 !important;
            font-weight: bold;
        }
        .result-negative {
            color: #ef4444 !important;
            font-weight: bold;
        }
        .confidence-high {
            color: #059669;
        }
        .confidence-medium {
            color: #d97706;
        }
        .confidence-low {
            color: #dc2626;
        }
        """
        
        with gr.Blocks(title="Deepfake Detection System", css=css, theme=gr.themes.Soft()) as app:
            gr.Markdown("""
            # 🔍 Deepfake Detection System
            ### Advanced AI-powered detection of manipulated images and videos
            Upload an image or video to analyze if it contains deepfakes or manipulated content.
            """)
            
            # Model info
            model_info = "🤖 **Current Model**: "
            if self.use_simple_model:
                model_info += "Traditional Machine Learning (OpenCV + Random Forest)"
            else:
                model_info += "Deep Learning (Neural Network)"
            
            gr.Markdown(model_info)
            
            with gr.Tab("📸 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil", 
                            label="Upload Image",
                            height=400
                        )
                        
                        with gr.Row():
                            image_button = gr.Button(
                                "🔍 Analyze Image", 
                                variant="primary",
                                size="lg"
                            )
                            clear_button = gr.Button(
                                "🗑️ Clear",
                                variant="secondary"
                            )
                    
                    with gr.Column(scale=1):
                        result_image = gr.Image(
                            label="Analysis Result",
                            height=400
                        )
                        
                        with gr.Row():
                            image_result = gr.Textbox(
                                label="Prediction Result", 
                                lines=2,
                                interactive=False
                            )
                            image_confidence = gr.Number(
                                label="Confidence Score",
                                precision=4,
                                interactive=False
                            )
                
                # Examples section
                gr.Markdown("### 📋 Try these examples:")
                gr.Examples(
                    examples=[],  # You can add example images here
                    inputs=image_input,
                    label="Sample Images"
                )
                
                image_button.click(
                    fn=self.predict_image,
                    inputs=[image_input],
                    outputs=[image_result, image_confidence, result_image]
                )
                
                clear_button.click(
                    fn=lambda: (None, "", 0.0, None),
                    outputs=[image_input, image_result, image_confidence, result_image]
                )
            
            with gr.Tab("🎥 Video Analysis"):
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="Upload Video")
                        video_button = gr.Button("🔍 Analyze Video", variant="primary")
                    
                    with gr.Column():
                        video_result = gr.Textbox(label="Result", lines=2)
                        video_confidence = gr.Number(label="Confidence Score", precision=4)
                        video_output = gr.Image(label="Video Analysis")
                
                if self.use_simple_model:
                    gr.Markdown("⚠️ **Note**: Video analysis is only available with deep learning models.")
                
                video_button.click(
                    fn=self.predict_video,
                    inputs=[video_input],
                    outputs=[video_result, video_confidence, video_output]
                )
            
            with gr.Tab("📊 Information"):
                gr.Markdown("""
                ## About This System
                
                This deepfake detection system uses advanced machine learning to analyze images and videos 
                for signs of artificial manipulation.
                
                ### 🔧 How It Works
                1. **Face Detection**: Automatically detects faces in uploaded media
                2. **Feature Analysis**: Analyzes facial features and patterns
                3. **AI Classification**: Uses trained models to classify content
                4. **Confidence Scoring**: Provides reliability metrics
                
                ### 📈 Model Performance
                - **Accuracy**: High accuracy on test datasets
                - **Speed**: Real-time analysis for images
                - **Robustness**: Works with various image qualities
                
                ### 🎯 Supported Formats
                - **Images**: JPG, PNG, BMP, TIFF
                - **Videos**: MP4, AVI, MOV (deep learning models only)
                
                ### ⚠️ Limitations
                - Performance may vary with image/video quality
                - Works best with clear, well-lit faces
                - May not detect all sophisticated manipulation techniques
                - Results should be used as guidance, not definitive proof
                
                ### 🔒 Privacy
                - Images are processed locally and not stored
                - No data is sent to external servers
                - Temporary files are automatically deleted
                """)
        
        return app

def main():
    """Main function to run the web application"""
    print("🚀 Starting Deepfake Detection Web Application...")
    
    # Check for available models
    deep_model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    simple_model_exists = os.path.exists('simple_deepfake_model.pkl')
    
    use_simple_model = False
    model_path = None
    
    # Determine which model to use
    if not deep_model_files and not simple_model_exists:
        print("❌ No trained models found!")
        print("Please train a model first:")
        print("- For deep learning: python train_model.py")
        print("- For simple model: python simple_detector.py")
        return
    
    if deep_model_files and simple_model_exists:
        print("Both model types available:")
        print("1. Deep Learning Models:", deep_model_files)
        print("2. Simple Traditional ML Model")
        
        choice = input("Choose model type (1 for deep learning, 2 for simple): ").strip()
        if choice == '2':
            use_simple_model = True
            print("Using Simple Traditional ML Model")
        else:
            if len(deep_model_files) == 1:
                model_path = deep_model_files[0]
            else:
                print("Available deep learning models:")
                for i, model in enumerate(deep_model_files):
                    print(f"{i+1}. {model}")
                model_choice = input("Enter model number: ").strip()
                try:
                    model_path = deep_model_files[int(model_choice) - 1]
                except (ValueError, IndexError):
                    model_path = deep_model_files[0]
            print(f"Using Deep Learning Model: {model_path}")
    
    elif simple_model_exists:
        use_simple_model = True
        print("Using Simple Traditional ML Model")
    
    else:
        model_path = deep_model_files[0]
        print(f"Using Deep Learning Model: {model_path}")
    
    # Create and launch web app
    try:
        app = DeepfakeWebApp(model_path=model_path, use_simple_model=use_simple_model)
        interface = app.create_interface()
        
        print("\n✅ Web application ready!")
        print("🌐 Access the app at: http://localhost:7860")
        print("🔗 Share link will be generated if share=True")
        print("⏹️  Press Ctrl+C to stop the server")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,  # Set to True to create a public link
            debug=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Error launching web application: {e}")
        print("Make sure Gradio is installed: pip install gradio")

if __name__ == "__main__":
    main()
