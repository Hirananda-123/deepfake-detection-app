#!/bin/bash

echo "🧪 Testing Deployment Setup..."
echo "================================"

# Test 1: Check if all required files exist
echo "📋 Checking required files..."

files=("app.py" "Procfile" "runtime.txt" "requirements-production.txt" "frontend/landing.html" "frontend/modern-index.html")

for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

# Test 2: Check if frontend directory has all files
echo ""
echo "📁 Checking frontend files..."
ls -la frontend/

# Test 3: Validate Python syntax
echo ""
echo "🐍 Validating Python syntax..."
python -m py_compile app.py && echo "✅ app.py syntax OK" || echo "❌ app.py syntax error"

# Test 4: Check requirements
echo ""
echo "📦 Checking requirements..."
if [ -f "requirements-production.txt" ]; then
    echo "✅ Production requirements ready"
    echo "📝 Dependencies:"
    head -5 requirements-production.txt
else
    echo "❌ Production requirements missing"
fi

echo ""
echo "🚀 Ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Choose deployment platform (Heroku, Railway, etc.)"
echo "2. Run: ./deploy.sh or deploy.bat"
echo "3. Follow platform-specific instructions"
echo ""
echo "Your website will have:"
echo "- Landing page at: https://your-app.herokuapp.com/"
echo "- Main app at: https://your-app.herokuapp.com/app"
echo "- API at: https://your-app.herokuapp.com/api/predict"
