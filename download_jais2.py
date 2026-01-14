#!/usr/bin/env python3
"""
Download script for Jais-2 model.

This script downloads Jais-2 models (8B or 70B) and caches them locally 
for offline use in the evaluation pipeline.

Usage:
    python scripts/download_jais2.py [model_name]
    
    Examples:
        python scripts/download_jais2.py                                    # Downloads 8B model (default)
        python scripts/download_jais2.py inceptionai/Jais-2-8B-Chat        # Downloads 8B model
        python scripts/download_jais2.py inceptionai/Jais-2-70B-Chat       # Downloads 70B model

Requirements:
    1. Hugging Face account with access to the Jais-2 models
    2. HUGGINGFACE_TOKEN in .env file (for gated repositories)
    3. Free disk space: ~16GB (8B model) or ~140GB (70B model)
    4. Internet connection for initial download

After download, the model can be used offline with local_files_only=True.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def check_requirements():
    """Check if all requirements are met."""
    print("=" * 80)
    print("JAIS-2 MODEL DOWNLOAD SCRIPT")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not Path("config.yaml").exists():
        print("[ERROR] Please run this script from the project root directory:")
        print("  python scripts/download_jais2.py")
        return False
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Check for Hugging Face token
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not hf_token:
        print("[ERROR] No Hugging Face token found!")
        print("[INFO] This model requires authentication. To fix:")
        print("  1. Get token from: https://huggingface.co/settings/tokens")
        print("  2. Accept access at: https://huggingface.co/inceptionai/Jais-2-8B-Chat")
        print("  3. Add to .env file: HUGGINGFACE_TOKEN=your_token_here")
        return False
    
    print("[OK] Requirements check passed")
    print(f"[INFO] Using token: {hf_token[:8]}...{hf_token[-8:]}")
    return True


def download_model():
    """Download the Jais-2 model using huggingface_hub."""
    try:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing required packages: {e}")
        print("[INFO] Install with: pip install transformers huggingface_hub torch")
        return False
    
    model_id = "inceptionai/Jais-2-8B-Chat"
    hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    
    print(f"\n[INFO] Downloading model: {model_id}")
    print(f"[INFO] Model size: {model_size}")
    print(f"[INFO] Estimated time: {download_time} depending on your connection...")
    if "70B" in model_size:
        print(f"[WARNING] ⚠️  This is a VERY LARGE model! Ensure you have:")
        print(f"  - At least 150GB free disk space")
        print(f"  - Stable internet connection")
        print(f"  - Sufficient time for download")
    
    try:
        # Download model files
        print(f"\n[STEP 1/3] Downloading model repository...")
        local_dir = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            token=hf_token
        )
        print(f"[OK] Model downloaded to: {local_dir}")
        
        # Verify tokenizer can load
        print(f"\n[STEP 2/3] Verifying tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )
        print(f"[OK] Tokenizer loaded successfully")
        print(f"[INFO] Vocab size: {tokenizer.vocab_size}")
        
        # Verify model can load (without moving to GPU to save time)
        print(f"\n[STEP 3/3] Verifying model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=torch.float16,  # Faster loading for verification
            device_map="cpu"  # Keep on CPU for verification
        )
        print(f"[OK] Model loaded successfully")
        print(f"[INFO] Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
        
        # Clean up model from memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n" + "=" * 80)
        print("DOWNLOAD COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"✅ Model: {model_id}")
        print(f"✅ Location: {local_dir}")
        print(f"✅ Ready for offline use")
        print(f"\nYou can now run the evaluation pipeline with:")
        print(f"  python scripts/main.py")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n[ERROR] Download failed: {e}")
        
        # Provide specific help based on error type
        if "403" in error_msg or "forbidden" in error_msg.lower():
            print(f"\n[HELP] Access denied - you need repository access:")
            print(f"  1. Visit: https://huggingface.co/inceptionai/Jais-2-8B-Chat")
            print(f"  2. Click 'Agree and access repository'")
            print(f"  3. Wait 1-5 minutes for access to propagate")
            print(f"  4. Re-run this script")
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            print(f"\n[HELP] Authentication failed:")
            print(f"  1. Check your token: https://huggingface.co/settings/tokens")
            print(f"  2. Ensure token has 'Read' permission")
            print(f"  3. Update HUGGINGFACE_TOKEN in .env file")
        else:
            print(f"\n[HELP] For other issues:")
            print(f"  1. Check internet connection")
            print(f"  2. Verify disk space (~16GB required)")
            print(f"  3. Try: huggingface-cli login")
        
        return False


def check_existing_model():
    """Check if model is already downloaded."""
    try:
        from transformers import AutoTokenizer
        
        model_id = "inceptionai/Jais-2-8B-Chat"
        
        # Try to load tokenizer in offline mode
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True
        )
        
        print(f"[INFO] Model already cached: {model_id}")
        print(f"[INFO] Vocab size: {tokenizer.vocab_size}")
        
        return True
        
    except Exception:
        return False


def main():
    """Main download process."""
    
    if not check_requirements():
        sys.exit(1)
    
    # Check if model is already downloaded
    if check_existing_model():
        print(f"\n[SKIP] Model is already downloaded and cached.")
        print(f"[INFO] You can run the evaluation pipeline with:")
        print(f"  python scripts/main.py")
        return
    
    # Download the model
    if download_model():
        print(f"\n[SUCCESS] Jais-2 model is ready for offline use!")
    else:
        print(f"\n[FAILURE] Download failed. Please fix the issues above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()