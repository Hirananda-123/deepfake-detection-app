#!/usr/bin/env python3
"""
Deployment Status Checker for Deepfake Detection Application
Checks current deployment readiness and provides next steps
"""

import os
import subprocess
import sys
from pathlib import Path

def check_command_exists(command):
    """Check if a command exists in the system"""
    try:
        subprocess.run([command, "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_git_status():
    """Check Git installation and repository status"""
    print("🔍 Checking Git status...")
    
    if not check_command_exists("git"):
        return {
            "installed": False,
            "status": "❌ Git is not installed",
            "action": "Install Git from https://git-scm.com/"
        }
    
    # Check if git repo exists
    if os.path.exists(".git"):
        try:
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return {
                    "installed": True,
                    "repo_exists": True,
                    "status": "✅ Git repository exists",
                    "uncommitted_changes": bool(result.stdout.strip())
                }
        except:
            pass
    
    return {
        "installed": True,
        "repo_exists": False,
        "status": "⚠️ Git installed but no repository initialized",
        "action": "Run 'git init' to initialize repository"
    }

def check_heroku_status():
    """Check Heroku CLI and app status"""
    print("🔍 Checking Heroku status...")
    
    if not check_command_exists("heroku"):
        return {
            "cli_installed": False,
            "status": "❌ Heroku CLI not installed",
            "action": "Install from https://devcenter.heroku.com/articles/heroku-cli"
        }
    
    try:
        # Check if logged in
        result = subprocess.run(["heroku", "auth:whoami"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {
                "cli_installed": True,
                "logged_in": True,
                "user": result.stdout.strip(),
                "status": f"✅ Logged in as {result.stdout.strip()}"
            }
        else:
            return {
                "cli_installed": True,
                "logged_in": False,
                "status": "⚠️ Heroku CLI installed but not logged in",
                "action": "Run 'heroku login' to authenticate"
            }
    except:
        return {
            "cli_installed": True,
            "status": "⚠️ Could not check Heroku login status"
        }

def check_deployment_files():
    """Check if all deployment files exist and are correct"""
    print("🔍 Checking deployment files...")
    
    required_files = {
        "Procfile": "web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120",
        "runtime.txt": "python-3.13.5",
        "requirements.txt": "Flask",  # Should contain Flask
        "app.py": "Flask",  # Should contain Flask imports
    }
    
    results = {}
    
    for filename, expected_content in required_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if expected_content in content:
                    results[filename] = {"status": "✅ Exists and correct", "exists": True}
                else:
                    results[filename] = {"status": f"⚠️ Exists but may need updates", "exists": True}
            except:
                results[filename] = {"status": "❌ Exists but cannot read", "exists": True}
        else:
            results[filename] = {"status": "❌ Missing", "exists": False}
    
    return results

def check_python_environment():
    """Check Python and virtual environment status"""
    print("🔍 Checking Python environment...")
    
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    return {
        "version": python_version,
        "venv_active": venv_active,
        "status": f"✅ Python {python_version}" + (" (venv active)" if venv_active else " (no venv)")
    }

def suggest_next_steps(git_status, heroku_status, files_status):
    """Suggest next steps based on current status"""
    print("\n📋 RECOMMENDED NEXT STEPS:")
    print("=" * 50)
    
    steps = []
    
    # Git setup
    if not git_status.get("installed"):
        steps.append("1. 📥 Install Git from https://git-scm.com/")
        steps.append("2. 🔄 Restart your terminal after installation")
    elif not git_status.get("repo_exists"):
        steps.append("1. 🚀 Initialize Git repository: git init")
        steps.append("2. 📦 Add files: git add .")
        steps.append("3. 💾 Make first commit: git commit -m 'Initial commit'")
    
    # Heroku setup
    if not heroku_status.get("cli_installed"):
        steps.append("4. 📥 Install Heroku CLI from https://devcenter.heroku.com/articles/heroku-cli")
        steps.append("5. 🔄 Restart your terminal after installation")
    elif not heroku_status.get("logged_in"):
        steps.append("4. 🔑 Login to Heroku: heroku login")
        steps.append("5. 🚀 Create app: heroku create your-app-name")
        steps.append("6. 📤 Deploy: git push heroku main")
    
    # Alternative deployment options
    if not git_status.get("installed") or not heroku_status.get("cli_installed"):
        steps.append("\n🔄 ALTERNATIVE DEPLOYMENT OPTIONS:")
        steps.append("   • Railway: https://railway.app (GitHub integration)")
        steps.append("   • Render: https://render.com (GitHub integration)")
        steps.append("   • Vercel: https://vercel.com (GitHub integration)")
        steps.append("   • PythonAnywhere: https://pythonanywhere.com")
    
    for step in steps:
        print(step)

def main():
    """Main deployment status check"""
    print("🚀 DEEPFAKE DETECTION - DEPLOYMENT STATUS CHECK")
    print("=" * 60)
    
    # Check Python environment
    python_status = check_python_environment()
    print(f"🐍 Python: {python_status['status']}")
    
    # Check Git
    git_status = check_git_status()
    print(f"📁 Git: {git_status['status']}")
    
    # Check Heroku
    heroku_status = check_heroku_status()
    print(f"☁️ Heroku: {heroku_status['status']}")
    
    # Check deployment files
    files_status = check_deployment_files()
    print(f"\n📄 DEPLOYMENT FILES:")
    for filename, info in files_status.items():
        print(f"   {info['status']} - {filename}")
    
    # Overall readiness
    print(f"\n📊 DEPLOYMENT READINESS:")
    git_ready = git_status.get("installed", False) and git_status.get("repo_exists", False)
    heroku_ready = heroku_status.get("cli_installed", False) and heroku_status.get("logged_in", False)
    files_ready = all(info["exists"] for info in files_status.values())
    
    print(f"   {'✅' if git_ready else '❌'} Git Repository Ready")
    print(f"   {'✅' if heroku_ready else '❌'} Heroku Account Ready") 
    print(f"   {'✅' if files_ready else '❌'} Deployment Files Ready")
    
    overall_ready = git_ready and heroku_ready and files_ready
    print(f"\n🎯 OVERALL STATUS: {'🟢 READY TO DEPLOY!' if overall_ready else '🟡 SETUP REQUIRED'}")
    
    # Suggest next steps
    suggest_next_steps(git_status, heroku_status, files_status)
    
    print(f"\n💡 TIP: You can also run 'deploy.bat' for guided deployment setup!")

if __name__ == "__main__":
    main()
