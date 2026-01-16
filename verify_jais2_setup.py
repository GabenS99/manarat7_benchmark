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
import argparse
import yaml
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

def check_jais2_architecture(model_id):
    """Check if Jais-2 architecture is recognized."""
    print("\n🏗️ Checking Jais-2 architecture support...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_id)
        print(f"   ✅ Jais-2 architecture recognized: {config.model_type}")
        print(f"   ✅ Model: {model_id}")
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

def check_model_access(model_id):
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
        info = model_info(model_id, token=hf_token)
        print(f"   ✅ Repository accessible: {info.modelId}")
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "authentication" in error_msg:
            print("   ❌ Authentication required")
            print("   💡 Fix: huggingface-cli login")
        elif "403" in error_msg or "forbidden" in error_msg:
            print("   ❌ Repository access not granted")
            print(f"   💡 Fix: Visit https://huggingface.co/{model_id}")
            print("   💡      Click 'Agree and access repository'")
        else:
            print(f"   ❌ Access check failed: {type(e).__name__}")
        return False

def check_cache_status(model_id):
    """Check model cache status."""
    print("\n💾 Checking model cache status...")
    
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        print("   ℹ️ No HF cache directory found (expected for first run)")
        return True
    
    # Check for specific model cache
    model_cache_name = f"models--{model_id.replace('/', '--')}"
    model_cache_path = cache_dir / model_cache_name
    
    if model_cache_path.exists():
        size_gb = sum(f.stat().st_size for f in model_cache_path.rglob('*') if f.is_file()) / (1024**3)
        print(f"   ✅ Found {model_id} in cache: {size_gb:.1f}GB")
        
        # Check if model is complete
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            if tokenizer.vocab_size > 0:
                print(f"   ✅ Model appears complete (vocab_size: {tokenizer.vocab_size})")
            else:
                print(f"   ⚠️ Model cache incomplete (vocab_size: 0) - download may be corrupted")
        except Exception:
            print(f"   ⚠️ Cannot verify model completeness")
    else:
        print(f"   ℹ️ {model_id} not cached yet (will download on first use)")
    
    # Also show all Jais models in cache
    jais_cache = list(cache_dir.glob("models--*jais*")) + list(cache_dir.glob("models--*Jais*"))
    if jais_cache and len(jais_cache) > 1:
        print(f"   ℹ️ Found {len(jais_cache)} total Jais model(s) in cache")
    
    return True

def check_gpu_availability(model_id):
    """Check GPU availability for model inference."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)
            
            # Calculate total VRAM if multiple GPUs
            total_vram = sum(torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)) / (1024**3)
            
            print(f"   ✅ CUDA available: {gpu_count} GPU(s)")
            print(f"   ✅ Current device: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
            if gpu_count > 1:
                print(f"   ✅ Total VRAM: {total_vram:.1f}GB across {gpu_count} GPUs")
            
            # Check requirements based on model size
            is_70b = "70B" in model_id or "70b" in model_id.lower()
            if is_70b:
                required_vram = 140  # ~140GB for 70B model
                if total_vram >= required_vram:
                    print(f"   ✅ Sufficient VRAM for Jais-2-70B (~140GB required)")
                elif total_vram >= 70:
                    print(f"   ⚠️ Limited VRAM for 70B model - may need quantization or CPU offloading")
                else:
                    print(f"   ❌ Insufficient VRAM for 70B model - need ~140GB, have {total_vram:.1f}GB")
            else:
                required_vram = 16  # ~16GB for 8B model
                if gpu_memory >= required_vram:
                    print(f"   ✅ Sufficient VRAM for Jais-2-8B (~16GB required)")
                else:
                    print(f"   ⚠️ Limited VRAM - may need CPU fallback or smaller models")
            
            return True
        else:
            print("   ⚠️ CUDA not available - will use CPU (slower)")
            return True
    except ImportError:
        print("   ⚠️ PyTorch not available")
        return False

def run_quick_test(model_id):
    """Run a quick Jais-2 inference test."""
    print("\n🧪 Running quick inference test...")
    
    try:
        # Add scripts directory to path (predictors.py is in scripts/)
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from predictors import get_prediction_local_jais2
        
        is_70b = "70B" in model_id or "70b" in model_id.lower()
        if is_70b:
            print(f"   ℹ️ Testing 70B model - this may take a minute to load...")
        
        result = get_prediction_local_jais2(
            question="ما هي عاصمة الإمارات؟",
            model_version=model_id,
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

def get_model_from_config():
    """Get model name from config.yaml if available."""
    try:
        config_path = Path("config.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                models = config.get("prediction_models", {})
                jais2_models = models.get("local_jais2", [])
                if jais2_models:
                    return jais2_models[0]  # Return first model
    except Exception:
        pass
    return None


def main():
    """Run all verification checks."""
    parser = argparse.ArgumentParser(
        description="Verify Jais-2 model setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_jais2_setup.py                                    # Uses model from config.yaml (or 8B default)
  python verify_jais2_setup.py inceptionai/Jais-2-8B-Chat        # Tests 8B model
  python verify_jais2_setup.py inceptionai/Jais-2-70B-Chat      # Tests 70B model
        """
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Model name (e.g., inceptionai/Jais-2-70B-Chat). If not provided, reads from config.yaml or defaults to 8B."
    )
    
    args = parser.parse_args()
    
    # Determine which model to test
    if args.model_name:
        model_id = args.model_name
    else:
        # Try to get from config.yaml
        model_id = get_model_from_config()
        if not model_id:
            # Default to 8B if nothing specified
            model_id = "inceptionai/Jais-2-8B-Chat"
            print(f"[INFO] No model specified, defaulting to: {model_id}")
        else:
            print(f"[INFO] Using model from config.yaml: {model_id}")
    
    print("=" * 80)
    print("🔍 JAIS-2 SETUP VERIFICATION")
    print("=" * 80)
    print(f"Verifying Jais-2 model support for: {model_id}")
    
    checks = [
        ("Transformers Version", check_transformers_version),
        ("Jais-2 Architecture", lambda: check_jais2_architecture(model_id)),
        ("Authentication", check_authentication),
        ("Model Access", lambda: check_model_access(model_id)),
        ("Cache Status", lambda: check_cache_status(model_id)),
        ("GPU Availability", lambda: check_gpu_availability(model_id)),
        ("Quick Inference Test", lambda: run_quick_test(model_id)),
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
        print(f"   • Visit: https://huggingface.co/{model_id}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)