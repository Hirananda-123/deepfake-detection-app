@echo off
echo Starting Deepfake Detection API Server...
cd /d "C:\Users\815863\OneDrive - Cognizant\Desktop\Github Copilot\Deepfake detection"
call .\.venv\Scripts\activate.bat
echo Virtual environment activated
python test_env.py
echo Environment test completed
echo Starting Flask server...
python api_server.py
pause
