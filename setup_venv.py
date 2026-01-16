#!/usr/bin/python3
"""
IslamGPT Virtual Environment Setup Script
=========================================

This script automates the creation and setup of a virtual environment
for the IslamGPT project with all required dependencies.

Usage:
    python setup_venv.py

The script will:
1. Create a virtual environment named 'islamgpt_env'
2. Upgrade pip to the latest version
3. Install all requirements from requirements.txt
4. Provide activation instructions

Requirements:
- Python 3.8+ installed on your system
- Internet connection for downloading packages
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(command, description="", check=True):
    """Run a shell command with error handling."""
    print(f"[INFO] {description}")
    print(f"[CMD]  {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        if isinstance(command, str):
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(command, check=check, 
                                  capture_output=True, text=True)
        
        if result.stdout.strip():
            print(f"[OUT]  {result.stdout.strip()}")
        
        return result
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}")
        if e.stdout:
            print(f"[STDOUT] {e.stdout}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return None


def check_python_version():
    """Check if Python version is 3.8+."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"[ERROR] Python 3.8+ required. Current version: {version.major}.{version.minor}")
        sys.exit(1)
    
    print(f"[OK] Python {version.major}.{version.minor}.{version.micro} detected")


def get_activation_command():
    """Get the appropriate virtual environment activation command."""
    system = platform.system().lower()
    
    if system == "windows":
        return "islamgpt_env\\Scripts\\activate"
    else:  # macOS, Linux, etc.
        return "source islamgpt_env/bin/activate"


def main():
    """Main setup process."""
    print("="*80)
    print("IslamGPT Virtual Environment Setup")
    print("="*80)
    
    # Check Python version
    check_python_version()
    
    # Check if requirements.txt exists
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found in current directory!")
        print("[INFO] Please run this script from the project root directory.")
        sys.exit(1)
    
    print(f"[OK] Found requirements.txt ({requirements_file.stat().st_size} bytes)")
    
    # Remove existing virtual environment if it exists
    venv_path = Path("islamgpt_env")
    if venv_path.exists():
        print("[INFO] Removing existing virtual environment...")
        import shutil
        shutil.rmtree(venv_path)
    
    # Create virtual environment
    print("\n" + "-"*50)
    print("STEP 1: Creating Virtual Environment")
    print("-"*50)
    
    result = run_command([sys.executable, "-m", "venv", "islamgpt_env"],
                        "Creating virtual environment 'islamgpt_env'...")
    
    if not result:
        print("[ERROR] Failed to create virtual environment")
        sys.exit(1)
    
    print("[OK] Virtual environment created successfully")
    
    # Determine python executable path in venv
    system = platform.system().lower()
    if system == "windows":
        python_exe = venv_path / "Scripts" / "python.exe"
        pip_exe = venv_path / "Scripts" / "pip.exe"
    else:
        python_exe = venv_path / "bin" / "python"
        pip_exe = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("\n" + "-"*50)
    print("STEP 2: Upgrading pip")
    print("-"*50)
    
    result = run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"],
                        "Upgrading pip to latest version...")
    
    if not result:
        print("[WARNING] Failed to upgrade pip, continuing anyway...")
    else:
        print("[OK] pip upgraded successfully")
    
    # Install requirements
    print("\n" + "-"*50)
    print("STEP 3: Installing Requirements")
    print("-"*50)
    
    result = run_command([str(pip_exe), "install", "-r", "requirements.txt"],
                        "Installing packages from requirements.txt...")
    
    if not result:
        print("[ERROR] Failed to install requirements")
        sys.exit(1)
    
    print("[OK] All requirements installed successfully")
    
    # Verify installation
    print("\n" + "-"*50)
    print("STEP 4: Verifying Installation")
    print("-"*50)
    
    # Test key imports
    test_imports = [
        "import openai",
        "import google.generativeai as genai", 
        "import groq",
        "import mistralai",
        "import transformers",
        "import torch",
        "import yaml",
        "import openpyxl",
        "import pandas",
        "import requests",
        "from dotenv import load_dotenv"
    ]
    
    failed_imports = []
    for import_stmt in test_imports:
        result = run_command([str(python_exe), "-c", import_stmt],
                           f"Testing: {import_stmt}", check=False)
        if not result or result.returncode != 0:
            failed_imports.append(import_stmt)
    
    if failed_imports:
        print(f"[WARNING] {len(failed_imports)} import(s) failed:")
        for imp in failed_imports:
            print(f"  - {imp}")
        print("[INFO] This may be normal if some optional dependencies are not needed")
    else:
        print("[OK] All key packages imported successfully")
    
    # Success message
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    
    activation_cmd = get_activation_command()
    
    print(f"""
🎉 Virtual environment setup completed successfully!

📁 Environment location: {venv_path.absolute()}
🔄 Activation command:   {activation_cmd}

Next steps:
1. Activate the virtual environment:
   {activation_cmd}

2. Create your .env file with API keys:
   cp .env.example .env
   # Edit .env with your actual API keys

3. Run the main script:
   python scripts/main.py

4. To deactivate when done:
   deactivate

📚 For more information, see README.md
""")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)