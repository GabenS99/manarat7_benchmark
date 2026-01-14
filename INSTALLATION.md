# Installation Guide

**For ALLaM and JAIS Model Providers**

Complete setup instructions for running the IslamGPT Evaluation Pipeline with your local models.

## 🎯 System Requirements for Providers

### Minimum Requirements
- **OS**: Linux (Ubuntu 18.04+) - **Recommended for GPU support**
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 32GB minimum (64GB recommended for larger models)
- **Storage**: 50GB+ free space (for model cache and results)
- **Internet**: Required for initial model download from Hugging Face

### GPU Requirements (REQUIRED)
- **ALLaM-34B**: NVIDIA GPU(s) with **68GB+ VRAM** (A100 80GB or 2x A100 40GB recommended)
- **Jais-2-70B**: NVIDIA GPU(s) with **140GB+ VRAM** (2x A100 80GB or 4x A100 40GB required)
- **ALLaM-7B**: NVIDIA GPU with **14GB+ VRAM** (RTX 3090/4090 recommended)
- **Jais-2-8B**: NVIDIA GPU with **16GB+ VRAM** (RTX 3090/4090 recommended)
- **CUDA**: Version 11.7+ (installed automatically with PyTorch)
- **Driver**: NVIDIA driver 450.80.02+ (check: `nvidia-smi`)

## 🚀 Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# 1. Clone repository
git clone <repository-url>
cd for_allam_operators

# 2. Run automated setup
python3 setup_venv.py

# 3. Verify installation
python3 verify_setup.py
```

The automated setup will:
- Create virtual environment (`islamgpt_env`)
- Install all dependencies from `requirements.txt`
- Verify package installations
- Provide next steps

### Method 2: Manual Setup

```bash
# 1. Clone repository
git clone <repository-url>
cd for_allam_operators

# 2. Create virtual environment
python3 -m venv islamgpt_env

# 3. Activate environment
source islamgpt_env/bin/activate  # Linux/macOS
# OR
islamgpt_env\Scripts\activate     # Windows

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Verify installation
python verify_setup.py
```

## 🔑 Hugging Face Configuration (REQUIRED)

### Step 1: Get Hugging Face Token

1. Visit: https://huggingface.co/settings/tokens
2. Create a new token with **"Read"** permissions
3. Copy the token (starts with `hf_...`)

### Step 2: Configure Environment

```bash
# Copy environment template
cp env_template.txt .env

# Add your Hugging Face token (REQUIRED for model access)
echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env
```

### Step 3: Request Model Access

**For ALLaM Providers:**
- Visit: https://huggingface.co/ALLaM-AI/ALLaM-7B-Instruct-preview
- Click **"Agree and access repository"**
- Wait 1-2 minutes for access approval

**For JAIS Providers:**
- Visit: https://huggingface.co/inceptionai/Jais-2-8B-Chat
- Click **"Agree and access repository"**
- Wait 1-2 minutes for access approval

> **Note**: Cloud API keys (OpenAI, Gemini, etc.) are **optional** and only needed if you want to compare your model against cloud models.

## 🤖 Provider Model Setup

### ALLaM Provider Setup (ALLaM-34B)

1. **Locate Your Model Directory**:
   - ALLaM-34B is a closed model (not on Hugging Face)
   - Find the directory containing your model files
   - Verify it contains: `config.json`, `tokenizer.json`, and model weights

2. **Verify GPU Requirements**:
   ```bash
   nvidia-smi
   # ALLaM-34B requires ~68GB VRAM (or multi-GPU setup)
   # Check available VRAM matches requirements
   ```

3. **Configure Model Path in config.yaml**:
   ```yaml
   prediction_models:
     local_allam: ["/absolute/path/to/your/allam-34b-model"]
     # Example: local_allam: ["/home/user/models/allam-34b"]
     # OR relative: local_allam: ["./models/allam-34b"]
     
     # IMPORTANT: Comment out other models:
     # local_jais: [jais_13b]
     # local_jais2: [inceptionai/Jais-2-8B-Chat]
   ```

4. **Test Model Loading**:
   ```bash
   python3 test_allam_local.py
   ```
   
   Expected output:
   - ✅ Model loads successfully
   - ✅ Generates response
   - ✅ GPU memory usage shown

5. **Configure for Evaluation**:
   ```yaml
   prediction_parameters:
     max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
     batch_save_size: 25      # Save progress frequently
   ```

### JAIS Provider Setup (Jais-2-70B)

1. **Install Bleeding-Edge Transformers** (REQUIRED):
   ```bash
   # Jais-2-70B REQUIRES transformers 5.0+ (bleeding-edge)
   pip install --upgrade git+https://github.com/huggingface/transformers.git
   ```

2. **Verify Transformers Version**:
   ```bash
   python3 -c "import transformers; print(transformers.__version__)"
   # Should show: 5.0.0.dev0 or similar (NOT 4.x.x)
   # If it shows 4.x.x, the upgrade failed - retry the upgrade command
   ```

3. **Verify GPU Requirements**:
   ```bash
   nvidia-smi
   # Jais-2-70B requires ~140GB VRAM (or multi-GPU setup)
   # Check available VRAM matches requirements
   ```

4. **Configure Hugging Face Token**:
   ```bash
   echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env
   ```

5. **Request Model Access**:
   - Visit: https://huggingface.co/inceptionai/Jais-2-70B-Chat
   - Click "Agree and access repository"
   - Wait 1-2 minutes for access approval

6. **Configure Model in config.yaml**:
   ```yaml
   prediction_models:
     local_jais2: [inceptionai/Jais-2-70B-Chat]
     # IMPORTANT: Comment out other models:
     # local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]
     # local_jais: [jais_13b]
   ```

7. **Test Model Setup**:
   ```bash
   python3 verify_jais2_setup.py
   ```
   
   Expected: **7/7 checks passed** ✅

8. **Configure for Evaluation**:
   ```yaml
   prediction_parameters:
     max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
     batch_save_size: 25      # Save progress frequently
   ```

## 🔧 Provider Configuration

### Basic Configuration (Provider Models)

**For ALLaM:**
```yaml
prediction_models:
  local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]

prediction_parameters:
  temperature: 0
  max_tokens: 2000
  batch_save_size: 25
  checkpoint_enabled: true
  max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
```

**For JAIS:**
```yaml
prediction_models:
  local_jais2: [inceptionai/Jais-2-8B-Chat]  # OR local_jais: [jais_13b]

prediction_parameters:
  temperature: 0
  max_tokens: 2000
  batch_save_size: 25
  checkpoint_enabled: true
  max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
```

### Advanced Configuration (Optional: Add Cloud Models)

If you want to compare your model against cloud models:

```yaml
prediction_models:
  local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]  # Your model
  # Optional: Add cloud models for comparison
  openai: [gpt-4o]
  gemini: [gemini-pro]
```

**Note**: Cloud models require additional API keys (see `env_template.txt`).

### Configuration Options

- **Data Selection**: Choose question types and difficulty levels
- **Batch Processing**: Configure checkpointing and resumption
- **Output Formats**: JSON, Excel, statistics
- **Performance**: Optimize for your GPU hardware

## ✅ Verification

### Quick Verification

```bash
python3 verify_setup.py
```

Expected output:
```
================================================================================
VERIFICATION SUMMARY
================================================================================
✅ PASS Python Version
✅ PASS Required Packages  
✅ PASS Configuration File
✅ PASS Environment Variables
✅ PASS Data Structure
✅ PASS Basic Functionality

📊 Overall: 6/6 tests passed

🎉 All tests passed! Your IslamGPT setup is ready.
```

### Component-Specific Tests

```bash
# Test specific models
python3 test_allam_local.py      # ALLaM model
python3 verify_jais2_setup.py    # Jais-2 model

# Test API connections (requires API keys)
python3 -c "
from scripts.clients import client_openai
print('OpenAI client:', client_openai is not None)
"
```

## 🚨 Provider Troubleshooting

### Common Provider Issues

#### 1. Python Version Errors
```bash
# Error: Python 3.8+ required
# Solution: Install newer Python
sudo apt update && sudo apt install python3.9  # Ubuntu
brew install python@3.9                        # macOS
```

#### 2. Package Installation Failures
```bash
# Error: Failed to install torch/transformers
# Solution: Update pip and retry
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### 3. Virtual Environment Issues
```bash
# Error: venv module not found
# Solution: Install venv
sudo apt install python3-venv  # Ubuntu
```

#### 4. Permission Errors
```bash
# Error: Permission denied
# Solution: Use user installation
pip install --user -r requirements.txt
```

### Hugging Face Authentication Issues

#### 1. Invalid Token
```bash
# Check token format
cat .env | grep HUGGINGFACE_TOKEN
# Should start with: hf_

# Verify token works
huggingface-cli whoami
# Should show your username
```

#### 2. Model Access Denied
```bash
# Check if you have access
huggingface-cli scan-cache | grep -i allam  # For ALLaM
huggingface-cli scan-cache | grep -i jais   # For JAIS

# Re-request access on Hugging Face website
# Wait 1-2 minutes after approval
```

#### 3. Authentication Errors
```bash
# Re-login if needed
huggingface-cli login

# Or set token directly
export HUGGINGFACE_TOKEN=hf_your_token_here
```

### Provider Model Issues

#### 1. GPU Out of Memory (OOM)
```bash
# Check available GPU memory
nvidia-smi

# Solution: Ensure single worker in config.yaml
prediction_parameters:
  max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
```

#### 2. Model Won't Load
```bash
# Test model loading
python3 test_allam_local.py      # For ALLaM
python3 verify_jais2_setup.py   # For Jais-2

# Check GPU detection
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Should print: CUDA available: True
```

#### 3. Model Download Failures
```bash
# Check internet connection
ping huggingface.co

# Manual model download test
python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('ALLaM-AI/ALLaM-7B-Instruct-preview')
print('Model downloaded successfully')
"
```

#### 4. Transformers Version Conflicts (Jais-2 Only)
```bash
# Jais-2 REQUIRES bleeding-edge transformers
pip install --upgrade git+https://github.com/huggingface/transformers.git

# Verify version
python3 -c "import transformers; print(transformers.__version__)"
# Should show: 5.0.0.dev0 or similar (NOT 4.x.x)

# ALLaM uses stable version (already installed)
# No upgrade needed for ALLaM
```

## 🔍 Advanced Setup

### Docker Installation (Optional)

For containerized deployment:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python3", "scripts/main.py"]
```

### Cluster Deployment

For high-performance computing:

```bash
# Install MPI support
pip install mpi4py

# Run with multiple GPUs
mpirun -n 4 python3 scripts/main.py
```

### Custom Model Integration

To add custom models:

1. Create predictor function in `scripts/predictors.py`
2. Add model configuration to `config.yaml`
3. Update model mapping in `scripts/main.py`

## 📊 Provider Performance Optimization

### Hardware Recommendations

| Model | CPU | RAM | GPU | Storage | Expected Speed |
|-------|-----|-----|-----|---------|----------------|
| **ALLaM-34B** | 16+ cores | 128GB | A100 80GB or 2x A100 40GB | 200GB SSD | ~5-8s/question |
| **Jais-2-70B** | 32+ cores | 256GB | 2x A100 80GB or 4x A100 40GB | 500GB SSD | ~8-12s/question |
| ALLaM-7B | 8+ cores | 32GB | RTX 3090/4090 (24GB) | 100GB SSD | ~3-5s/question |
| Jais-2-8B | 8+ cores | 32GB | RTX 3090/4090 (24GB) | 100GB SSD | ~4-6s/question |

### Performance Tips

1. **Always use `max_parallel_workers: 1`** for GPU models
2. **Enable checkpointing** for long runs (resume if interrupted)
3. **First run is slow** (model download), subsequent runs use cache
4. **Monitor GPU memory**: `watch -n 1 nvidia-smi` during evaluation
5. **Close other GPU processes** before running evaluation

### Expected Performance

- **Model loading**: 30-60 seconds (after first download)
- **Per question**: 3-6 seconds (depending on model)
- **Full evaluation**: 2-4 hours for 2,000+ questions
- **Results saved**: Every 25 questions (configurable)

## 🆘 Provider Support

### Quick Help Checklist

1. **GPU Working?**
   ```bash
   nvidia-smi  # Should show your GPU
   python3 -c "import torch; print(torch.cuda.is_available())"  # Should be True
   ```

2. **Model Access?**
   ```bash
   huggingface-cli whoami  # Should show your username
   ```

3. **Model Loads?**
   ```bash
   python3 test_allam_local.py      # For ALLaM
   python3 verify_jais2_setup.py   # For Jais-2
   ```

4. **Configuration Correct?**
   - Check `config.yaml`: Model name matches your provider
   - Check `max_parallel_workers: 1` (CRITICAL)
   - Check `.env`: `HUGGINGFACE_TOKEN` is set

### Detailed Documentation

For additional help:
- **README.md** - Main documentation with quick start guide
- **INSTALLATION.md** - This file (complete installation guide)
- **config.yaml** - Configuration file with detailed inline comments
- **verify_setup.py** - Automated setup verification script

---

## ✅ Installation Complete! 🎉

**Next Steps for Providers:**

1. ✅ **Verify setup**: `python3 verify_setup.py` (should show 6/6 tests passed)
2. ✅ **Test your model**: 
   - ALLaM: `python3 test_allam_local.py`
   - Jais-2: `python3 verify_jais2_setup.py`
3. ✅ **Configure model**: Edit `config.yaml` with your model
4. ✅ **Run evaluation**: `python3 scripts/main.py`
5. ✅ **Review results**: Check `results/predictions/` and `results/statistics/`

**Your model is ready for benchmarking! 🚀**