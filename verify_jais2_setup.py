#!/usr/bin/env python3
"""
Jais-2 Setup Verification Script

This script verifies that all requirements for Jais-2 support are properly configured:
1. Transformers version (bleeding-edge required)
2. Hugging Face authentication
3. Model architecture support
4. Repository access

Run this after installation to ensure everything is working correctly.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_transformers_version():
    """Check if transformers version supports Jais-2."""
    print("\n🔍 Checking Transformers version...")
    
    try:
        import transformers
        version = transformers.__version__
        print(f"   Version: {version}")
        
        # Check if it's the dev version (required for Jais-2)
        if "dev" in version or version.startswith("5."):
            print("   ✅ Bleeding-edge transformers detected (Jais-2 supported)")
            return True
        else:
            print(f"   ❌ Stable version detected. Need bleeding-edge for Jais-2 support")
            print(f"   💡 Fix: pip install --upgrade git+https://github.com/huggingface/transformers.git")
            return False
    except ImportError:
        print("   ❌ Transformers not installed")
        return False

def check_jais2_architecture():
    """Check if Jais-2 architecture is recognized."""
    print("\n🏗️ Checking Jais-2 architecture support...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("inceptionai/Jais-2-8B-Chat")
        print(f"   ✅ Jais-2 architecture recognized: {config.model_type}")
        return True
    except Exception as e:
        print(f"   ❌ Jais-2 architecture not recognized: {type(e).__name__}")
        print(f"   💡 Error: {str(e)[:100]}...")
        return False

def check_authentication():
    """Check Hugging Face authentication status."""
    print("\n🔐 Checking Hugging Face authentication...")
    
    # Load .env file (same as clients.py does)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass  # dotenv not available, continue with environment variables only
    
    # Check environment variables (now includes .env file)
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if hf_token:
        print(f"   ✅ HF token found in environment (length: {len(hf_token)})")
        token_status = True
    else:
        print("   ⚠️ No HF token in environment variables")
        token_status = False
    
    # Check HF CLI authentication
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"   ✅ HF CLI authenticated as: {username}")
            return True
        else:
            print("   ⚠️ HF CLI not authenticated")
            return token_status
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ⚠️ HF CLI not available or timeout")
        return token_status

def check_model_access():
    """Check if user has access to Jais-2 model repository."""
    print("\n🔑 Checking model repository access...")
    
    try:
        from huggingface_hub import model_info
        # Load .env to get token if available
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        info = model_info("inceptionai/Jais-2-8B-Chat", token=hf_token)
        print(f"   ✅ Repository accessible: {info.modelId}")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "authentication" in error_msg:
            print("   ❌ Authentication required")
            print("   💡 Fix: huggingface-cli login")
        elif "403" in error_msg or "forbidden" in error_msg:
            print("   ❌ Repository access not granted")
            print("   💡 Fix: Visit https://huggingface.co/inceptionai/Jais-2-8B-Chat")
            print("   💡      Click 'Agree and access repository'")
        else:
            print(f"   ❌ Access check failed: {type(e).__name__}")
        return False

def check_cache_status():
    """Check model cache status."""
    print("\n💾 Checking model cache status...")
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        print("   ℹ️ No HF cache directory found (expected for first run)")
        return True
    
    # Hugging Face uses "models--org--model-name" format for cache directories
    # Check for both lowercase and mixed case patterns
    jais_cache = list(cache_dir.glob("models--*jais*")) + list(cache_dir.glob("models--*Jais*"))
    if jais_cache:
        print(f"   ✅ Found {len(jais_cache)} Jais model(s) in cache:")
        for cache_path in jais_cache:
            size_mb = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file()) / (1024*1024)
            print(f"      - {cache_path.name} (~{size_mb:.0f}MB)")
    else:
        print("   ℹ️ No Jais models cached yet (will download on first use)")
    
    return True

def check_gpu_availability():
    """Check GPU availability for model inference."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   ✅ Current device: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            
            if gpu_memory >= 15:
                print("   ✅ Sufficient VRAM for Jais-2-8B (~16GB required)")
            else:
                print("   ⚠️ Limited VRAM - may need CPU fallback or smaller models")
            
            return True
        else:
            print("   ⚠️ CUDA not available - will use CPU (slower)")
            return True
    except ImportError:
        print("   ⚠️ PyTorch not available")
        return False

def run_quick_test():
    """Run a quick Jais-2 inference test."""
    print("\n🧪 Running quick inference test...")
    
    try:
        # Add scripts directory to path (predictors.py is in scripts/)
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from predictors import get_prediction_local_jais2
        
        result = get_prediction_local_jais2(
            question="ما هي عاصمة الإمارات؟",
            model_version="inceptionai/Jais-2-8B-Chat",
            question_type="KNOW",
            temperature=0,
            max_tokens=30
        )
        
        if result and result.get("text"):
            response = result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]
            print(f"   ✅ Inference successful!")
            print(f"   📝 Response: {response}")
            print(f"   📊 Tokens: {result.get('tokens', {})}")
            return True
        else:
            print("   ❌ Inference failed - no response generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Inference test failed: {type(e).__name__}")
        print(f"   💡 Error: {str(e)[:100]}...")
        return False

def main():
    """Run all verification checks."""
    print("=" * 80)
    print("🔍 JAIS-2 SETUP VERIFICATION")
    print("=" * 80)
    print("Verifying Jais-2 model support configuration...")
    
    checks = [
        ("Transformers Version", check_transformers_version),
        ("Jais-2 Architecture", check_jais2_architecture),
        ("Authentication", check_authentication),
        ("Model Access", check_model_access),
        ("Cache Status", check_cache_status),
        ("GPU Availability", check_gpu_availability),
        ("Quick Inference Test", run_quick_test),
    ]
    
    results = {}
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
        except Exception as e:
            print(f"   ❌ Check failed with error: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {check_name}")
    
    print(f"\n📊 Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 SUCCESS: Jais-2 setup is fully functional!")
        print("\n✅ Ready to use:")
        print("   • python scripts/main.py")
        print("   • Jais-2 models in evaluation pipeline")
        return 0
    elif passed >= 5:  # Core functionality working
        print("\n⚠️ MOSTLY WORKING: Core functionality available")
        print("   Some optional features may not work optimally")
        return 0
    else:
        print("\n❌ SETUP ISSUES: Please fix the failed checks above")
        print("\n💡 Common fixes:")
        print("   • pip install --upgrade git+https://github.com/huggingface/transformers.git")
        print("   • huggingface-cli login")
        print("   • Visit: https://huggingface.co/inceptionai/Jais-2-8B-Chat")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)