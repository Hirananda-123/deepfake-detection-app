#!/usr/bin/env python3
"""
Comprehensive test script to identify and fix issues in the deepfake detection application
"""

import os
import sys
import json
import traceback
from pathlib import Path

def test_file_syntax(file_path, file_type):
    """Test syntax of various file types"""
    try:
        if file_type == 'python':
            with open(file_path, 'r', encoding='utf-8') as f:
                compile(f.read(), file_path, 'exec')
            return True, "✅ Python syntax OK"
            
        elif file_type == 'javascript':
            # Basic JS syntax checks
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for balanced braces, parentheses, brackets
            if content.count('{') != content.count('}'):
                return False, f"❌ Unmatched braces: {content.count('{')} open, {content.count('}')} close"
            if content.count('(') != content.count(')'):
                return False, f"❌ Unmatched parentheses: {content.count('(')} open, {content.count(')')} close"
            if content.count('[') != content.count(']'):
                return False, f"❌ Unmatched brackets: {content.count('[')} open, {content.count(']')} close"
            
            return True, "✅ JavaScript syntax appears OK"
            
        elif file_type == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True, "✅ JSON syntax OK"
            
        return True, "✅ File exists"
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def test_imports():
    """Test all Python module imports"""
    print("\n📦 Testing Python imports...")
    
    modules_to_test = [
        'flask',
        'flask_cors', 
        'cv2',
        'numpy',
        'sklearn',
        'PIL',
        'pickle'
    ]
    
    results = []
    for module in modules_to_test:
        try:
            __import__(module)
            results.append(f"✅ {module}")
        except ImportError as e:
            results.append(f"❌ {module}: {e}")
    
    return results

def test_project_files():
    """Test all project files for syntax errors"""
    print("\n📁 Testing project files...")
    
    files_to_test = [
        ('app.py', 'python'),
        ('api_server.py', 'python'), 
        ('simple_detector.py', 'python'),
        ('advanced_detector.py', 'python'),
        ('frontend/modern-script.js', 'javascript'),
        ('frontend/modern-index.html', 'html'),
        ('frontend/landing.html', 'html'),
        ('requirements.txt', 'text'),
        ('Procfile', 'text'),
        ('runtime.txt', 'text')
    ]
    
    results = []
    for file_path, file_type in files_to_test:
        if os.path.exists(file_path):
            success, message = test_file_syntax(file_path, file_type)
            results.append(f"{message} - {file_path}")
        else:
            results.append(f"❌ File missing: {file_path}")
    
    return results

def test_directory_structure():
    """Test if all required directories exist"""
    print("\n📂 Testing directory structure...")
    
    required_dirs = [
        'frontend',
        'temp_uploads',
        '.venv'
    ]
    
    results = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            results.append(f"✅ Directory exists: {dir_name}")
        else:
            results.append(f"❌ Directory missing: {dir_name}")
            
    return results

def test_application_startup():
    """Test if the application can start without errors"""
    print("\n🚀 Testing application startup...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Test importing app module
        import app
        
        # Test detector initialization
        app.init_detector()
        
        # Test Flask app existence
        if hasattr(app, 'app'):
            return ["✅ Flask application object exists", 
                   f"✅ Detector type: {app.detector_type}",
                   "✅ Application startup test passed"]
        else:
            return ["❌ Flask application object missing"]
            
    except Exception as e:
        return [f"❌ Application startup failed: {str(e)}"]

def main():
    """Run all tests and provide summary"""
    print("🔍 DEEPFAKE DETECTION APPLICATION - ERROR DIAGNOSIS")
    print("=" * 60)
    
    all_results = []
    
    # Test imports
    import_results = test_imports()
    all_results.extend(import_results)
    
    # Test project files
    file_results = test_project_files()
    all_results.extend(file_results)
    
    # Test directory structure
    dir_results = test_directory_structure()
    all_results.extend(dir_results)
    
    # Test application startup
    app_results = test_application_startup()
    all_results.extend(app_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    success_count = 0
    error_count = 0
    
    for result in all_results:
        print(result)
        if result.startswith("✅"):
            success_count += 1
        elif result.startswith("❌"):
            error_count += 1
    
    print(f"\n📈 SUMMARY: {success_count} passed, {error_count} failed")
    
    if error_count == 0:
        print("\n🎉 ALL TESTS PASSED! Your application is ready for deployment.")
    else:
        print(f"\n⚠️  Found {error_count} issues that need to be fixed.")
        
    return error_count == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Test script error: {e}")
        traceback.print_exc()
        sys.exit(1)
