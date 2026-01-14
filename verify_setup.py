#!/usr/bin/env python3
"""
IslamGPT Setup Verification Script
==================================

This script verifies that your IslamGPT installation is working correctly.

Usage:
    python verify_setup.py

The script will test:
1. Python version compatibility
2. Required packages installation  
3. Configuration file loading
4. API key configuration
5. Basic functionality
"""

import sys
import os
from pathlib import Path
import importlib


def check_python_version():
    """Check Python version compatibility."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_packages():
    """Check if required packages are installed."""
    print("\n🔍 Checking required packages...")
    
    required_packages = [
        ("openai", "OpenAI API client"),
        ("google.generativeai", "Google Gemini client"), 
        ("groq", "Groq API client"),
        ("mistralai", "Mistral AI client"),
        ("transformers", "Hugging Face Transformers"),
        ("torch", "PyTorch"),
        ("yaml", "YAML parser"),
        ("openpyxl", "Excel file support"),
        ("pandas", "Data manipulation"),
        ("requests", "HTTP requests"),
        ("dotenv", "Environment variables")
    ]
    
    success_count = 0
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package:<20} - {description}")
            success_count += 1
        except ImportError:
            print(f"❌ {package:<20} - {description} (NOT FOUND)")
    
    print(f"\n📊 {success_count}/{len(required_packages)} packages available")
    return success_count == len(required_packages)


def check_config_file():
    """Check if config.yaml exists and is loadable."""
    print("\n🔍 Checking configuration file...")
    
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("❌ config.yaml not found")
        return False
    
    try:
        # Test loading with the project's config loader
        sys.path.insert(0, "scripts")
        from config_loader import load_config
        
        config = load_config("config.yaml")
        print("✅ config.yaml loaded successfully")
        
        # Basic validation
        if hasattr(config, 'prediction_data') and hasattr(config, 'models'):
            print("✅ Configuration structure is valid")
            return True
        else:
            print("❌ Configuration structure is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Error loading config.yaml: {e}")
        return False


def check_env_file():
    """Check environment variables setup."""
    print("\n🔍 Checking environment configuration...")
    
    # Check if env template exists
    env_template = Path("env_template.txt")
    env_file = Path(".env")
    
    if not env_template.exists():
        print("❌ env_template.txt not found")
        return False
    
    print("✅ env_template.txt found")
    
    if not env_file.exists():
        print("⚠️  .env file not found (copy env_template.txt to .env)")
        return False
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_keys = [
            ("OPENAI_API_KEY", "OpenAI"),
            ("GEMINI_API_KEY", "Google Gemini"),
            ("GROQ_API_KEY", "Groq"),
            ("DEEPSEEK_API_KEY", "DeepSeek"),
            ("MISTRAIL_API_KEY", "Mistral"),
            ("FANAR_API_KEY", "Fanar")
        ]
        
        configured_keys = 0
        for key, name in api_keys:
            value = os.getenv(key)
            if value and value != f"your_{key.lower()}_here":
                print(f"✅ {name:<15} - Configured")
                configured_keys += 1
            else:
                print(f"⚠️  {name:<15} - Not configured")
        
        print(f"\n📊 {configured_keys}/{len(api_keys)} API keys configured")
        return True
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False


def check_data_structure():
    """Check if data directories exist."""
    print("\n🔍 Checking data structure...")
    
    required_dirs = [
        "data/MCQ",
        "data/COMP", 
        "data/KNOW"
    ]
    
    existing_dirs = 0
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            # Check if directory has JSON files
            json_files = list(path.rglob("*.json"))
            print(f"✅ {dir_path:<12} - {len(json_files)} JSON files found")
            existing_dirs += 1
        else:
            print(f"❌ {dir_path:<12} - Not found")
    
    print(f"\n📊 {existing_dirs}/{len(required_dirs)} data directories found")
    return existing_dirs > 0


def run_basic_test():
    """Run a basic functionality test."""
    print("\n🔍 Running basic functionality test...")
    
    try:
        # Test configuration loading
        sys.path.insert(0, "scripts")
        from config_loader import load_config
        
        config = load_config("config.yaml")
        print("✅ Configuration loading works")
        
        # Test constants import
        from constants import QuestionType, Difficulty
        print("✅ Constants module works")
        
        # Test basic path operations
        all_files = config.get_all_json_files()
        print(f"✅ Found {len(all_files)} JSON files to process")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def main():
    """Main verification process."""
    print("="*80)
    print("IslamGPT Setup Verification")
    print("="*80)
    
    tests = [
        ("Python Version", check_python_version),
        ("Required Packages", check_packages), 
        ("Configuration File", check_config_file),
        ("Environment Variables", check_env_file),
        ("Data Structure", check_data_structure),
        ("Basic Functionality", run_basic_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your IslamGPT setup is ready.")
        print("\nNext steps:")
        print("1. Configure API keys in .env file (if not done)")
        print("2. Review config.yaml settings")  
        print("3. Run: python scripts/main.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix the issues above.")
        print("\nFor help, see INSTALLATION.md")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INFO] Verification interrupted by user")
        sys.exit(1)