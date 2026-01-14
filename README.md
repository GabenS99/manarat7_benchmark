# MANARAT7 BENCHMARK

**For ALLaM and JAIS Model Providers**

A comprehensive benchmarking pipeline for evaluating your Arabic language models on Islamic domain knowledge. This repository is designed for **model providers** to test and benchmark their local ALLaM and JAIS models.

## 🎯 Overview

This evaluation framework tests Arabic language models on Islamic knowledge using three question types:
- **MCQ** (Multiple Choice Questions)
- **COMP** (Comprehension/Text Analysis)
- **KNOW** (Knowledge Questions)

Each question type includes three difficulty levels: Beginner, Intermediate, and Advanced.

## 🚀 Quick Start for Providers

### Prerequisites
- Python 3.8+ 
- **GPU with 14GB+ VRAM** (required for local models)
- 32GB+ RAM (recommended for model loading)
- Internet connection (for initial model download from Hugging Face)

### ⚠️ CRITICAL: Transformers Version Incompatibility

**ALLaM and Jais-2 models require DIFFERENT transformers versions and CANNOT be used together:**

- **ALLaM Models**: Require `transformers>=4.45.0` (stable)
- **Jais-2 Models**: Require `transformers 5.0` (bleeding-edge)

**Solution**: Use separate virtual environments or choose only one model type per deployment.

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd manarat7_benchmark

# Automated setup (recommended)
python3 setup_venv.py

# Manual setup (alternative)
python3 -m venv islamgpt_env
source islamgpt_env/bin/activate  # Linux/macOS
# islamgpt_env\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure Hugging Face Access

```bash
# Copy environment template
cp env_template.txt .env

# Add your Hugging Face token (required for model access)
echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env
```

**Get your token**: https://huggingface.co/settings/tokens

> **Note**: Cloud API keys (OpenAI, Gemini, etc.) are optional and only needed if you want to compare against cloud models.

### 3. Configure Your Model

Edit `config.yaml` to specify which model to evaluate:

**For ALLaM Providers (ALLaM-34B):**
```yaml
prediction_models:
  local_allam: ["/path/to/your/allam-34b-model"]  # REPLACE with your actual model path
  # IMPORTANT: Comment out other local models:
  # local_jais: [jais_13b]
  # local_jais2: [inceptionai/Jais-2-8B-Chat]
```

**Model Path Configuration:**
- **Absolute path**: `/home/user/models/allam-34b` or `/data/models/allam-34b`
- **Relative path**: `./models/allam-34b` (if model is in project directory)
- **Path requirements**: The directory must contain:
  - `config.json` (model configuration)
  - `tokenizer.json` or `tokenizer_config.json` (tokenizer files)
  - Model weight files (`.bin`, `.safetensors`, or similar)
- **Verification**: Check that the path exists: `ls /path/to/your/allam-34b-model`

**For JAIS Providers (Jais-2-70B):**
```yaml
prediction_models:
  local_jais2: [inceptionai/Jais-2-70B-Chat]  # Jais-2-70B model
  # IMPORTANT: Comment out other local models:
  # local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]
  # local_jais: [jais_13b]
```

**Request model access** on Hugging Face:
- Jais-2-70B: https://huggingface.co/inceptionai/Jais-2-70B-Chat
- Click "Agree and access repository" and wait 1-2 minutes for approval

### 4. Verify Your Model Setup

**For ALLaM Providers (ALLaM-34B):**
```bash
# 1. Verify model path exists
ls /path/to/your/allam-34b-model

# 2. Test model loading
python3 test_allam_local.py
```
Expected: Model loads successfully and generates responses ✅

**For JAIS Providers (Jais-2-70B):**
```bash
# 1. Verify transformers upgrade (REQUIRED)
python3 -c "import transformers; print(transformers.__version__)"
# Should show: 5.0.0.dev0 or similar (NOT 4.x.x)

# 2. If not upgraded, upgrade now:
pip install --upgrade git+https://github.com/huggingface/transformers.git

# 3. Verify setup
python3 verify_jais2_setup.py
```
Expected: 7/7 checks passed ✅

### 5. Run Evaluation

```bash
# Activate environment
source islamgpt_env/bin/activate

# Run full evaluation pipeline
python3 scripts/main.py
```

The pipeline will:
1. Load your model into GPU memory
2. Process all questions (75 files, ~3,500+ questions)
3. Save predictions in `results/predictions/`
4. Generate performance statistics

## 🤖 Supported Provider Models

### ALLaM Models
| Model | Size | VRAM Required | Setup | Notes |
|-------|------|---------------|-------|-------|
| ALLaM-34B | 34B | ~68GB | Standard transformers (4.45+) | **Closed model - local path required** |
| ALLaM-7B-Instruct | 7B | ~14GB | Standard transformers (4.45+) | Available on Hugging Face |
| ALLaM-2-7B | 7B | ~14GB | Standard transformers (4.45+) | Available on Hugging Face |

### JAIS Models
| Model | Size | VRAM Required | Setup | Notes |
|-------|------|---------------|-------|-------|
| Jais-2-70B-Chat | 70B | ~140GB | **Bleeding-edge transformers** (5.0+) | **REQUIRED for evaluation** |
| JAIS-13B | 13B | ~26GB | Standard transformers (4.45+) | Legacy model |
| Jais-2-8B-Chat | 8B | ~16GB | **Bleeding-edge transformers** (5.0+) | Smaller variant |

> **Note**: Cloud models (OpenAI, Gemini, etc.) are optional and can be added for comparison. See [Advanced Configuration](#advanced-configuration) below.

## 🔧 Provider Configuration

### Model Selection (Primary)
```yaml
prediction_models:
  # For ALLaM providers:
  local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]
  
  # For JAIS providers:
  local_jais: [jais_13b]  # OR
  local_jais2: [inceptionai/Jais-2-8B-Chat]
```

### Evaluation Parameters (Optimized for Local Models)
```yaml
prediction_parameters:
  temperature: 0          # Deterministic output
  max_tokens: 2000       # Response length limit
  few_shots: false       # Enable/disable few-shot examples
  show_cot: false        # Chain of thought reasoning
  batch_save_size: 25    # Save every N questions
  checkpoint_enabled: true # Resume if interrupted
  max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
```

### Data Selection
```yaml
data_to_predict:
  question_types: [KNOW, COMP, MCQ]  # All question types
  difficulty_levels: [Beginner, Intermediate, Advanced]  # All levels
```

### Advanced Configuration (Optional: Add Cloud Models for Comparison)
```yaml
prediction_models:
  local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]  # Your model
  # Optional: Add cloud models for comparison
  # openai: [gpt-4o]
  # gemini: [gemini-pro]
```

## 📁 Output Structure

Results are saved in organized directories:

```
results/
├── predictions/           # Model predictions
│   ├── MCQ/
│   │   ├── A_beginner/   # JSON + Excel outputs
│   │   ├── B_intermediate/
│   │   └── C_advanced/
│   ├── COMP/
│   └── KNOW/
├── statistics/           # Performance metrics
└── evaluations/         # Evaluation results (if enabled)
```

## 🛠️ Provider-Specific Setup

### ALLaM Provider Setup (ALLaM-34B)

1. **Locate Your Model:**
   - ALLaM-34B is a closed model (not on Hugging Face)
   - Ensure you have the model files in a local directory
   - The directory should contain: `config.json`, `tokenizer.json`, and model weight files (`.bin` or `.safetensors`)

2. **Configure Model Path in config.yaml:**
   ```yaml
   prediction_models:
     local_allam: ["/absolute/path/to/your/allam-34b-model"]
     # OR if in project directory:
     # local_allam: ["./models/allam-34b"]
   ```

3. **Comment Out Other Models:**
   ```yaml
   prediction_models:
     local_allam: ["/path/to/your/allam-34b-model"]
     # local_jais: [jais_13b]  # COMMENTED OUT
     # local_jais2: [inceptionai/Jais-2-8B-Chat]  # COMMENTED OUT
   ```

4. **Verify GPU Requirements:**
   - ALLaM-34B requires **~68GB VRAM** (or multi-GPU setup)
   - Check available VRAM: `nvidia-smi`
   - Ensure you have sufficient GPU memory before running

5. **Test Your Model:**
   ```bash
   python3 test_allam_local.py
   ```
   
   Expected: Model loads successfully and generates responses ✅

6. **Optimize Configuration:**
   ```yaml
   prediction_parameters:
     max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
     batch_save_size: 25      # Save progress frequently
   ```

### JAIS Provider Setup (Jais-2-70B)

1. **Upgrade Transformers (REQUIRED):**
   ```bash
   # Jais-2-70B REQUIRES bleeding-edge transformers (5.0+)
   pip install --upgrade git+https://github.com/huggingface/transformers.git
   
   # Verify version
   python3 -c "import transformers; print(transformers.__version__)"
   # Should show: 5.0.0.dev0 or similar (NOT 4.x.x)
   ```

2. **Configure Hugging Face Token:**
   ```bash
   echo "HUGGINGFACE_TOKEN=hf_your_token_here" >> .env
   ```

3. **Request Model Access:**
   - Visit: https://huggingface.co/inceptionai/Jais-2-70B-Chat
   - Click "Agree and access repository"
   - Wait 1-2 minutes for access approval

4. **Configure Model in config.yaml:**
   ```yaml
   prediction_models:
     local_jais2: [inceptionai/Jais-2-70B-Chat]
     # IMPORTANT: Comment out other local models:
     # local_allam: [ALLaM-AI/ALLaM-7B-Instruct-preview]
     # local_jais: [jais_13b]
   ```

5. **Verify GPU Requirements:**
   - Jais-2-70B requires **~140GB VRAM** (or multi-GPU setup)
   - Check available VRAM: `nvidia-smi`
   - Ensure you have sufficient GPU memory before running

6. **Verify Setup:**
   ```bash
   python3 verify_jais2_setup.py
   ```
   
   Expected: 7/7 checks passed ✅

7. **Optimize Configuration:**
   ```yaml
   prediction_parameters:
     max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
     batch_save_size: 25      # Save progress frequently
   ```

### Batch Processing

For large datasets, enable batch processing:

```yaml
prediction_parameters:
  batch_save_size: 25      # Save every 25 questions
  checkpoint_enabled: true # Resume from interruptions
```

### Parallel Execution

Optimize performance with parallel processing:

```yaml
prediction_parameters:
  enable_parallel_processing: true
  max_parallel_workers: 5    # Adjust based on your system
```

## 📈 Performance Optimization for Providers

### GPU Memory Management
```yaml
prediction_parameters:
  max_parallel_workers: 1  # CRITICAL: Always use 1 for local GPU models
  # Multiple workers will cause OOM errors
```

### Model Caching
- First run: Downloads model (~14-16GB, 10-30 minutes)
- Subsequent runs: Uses cache (30-60 seconds to load)
- Cache location: `~/.cache/huggingface/hub/`

### Batch Processing
```yaml
prediction_parameters:
  batch_save_size: 25      # Save every 25 questions
  checkpoint_enabled: true # Resume if interrupted
```

### Expected Performance
- **ALLaM-34B**: ~5-8 seconds per question (larger model, more computation)
- **Jais-2-70B**: ~8-12 seconds per question (largest model, requires multi-GPU)
- **ALLaM-7B**: ~3-5 seconds per question
- **Jais-2-8B**: ~4-6 seconds per question
- **Full evaluation**: 3-6 hours for 2,000+ questions (depending on model size)

### Hardware Recommendations
| Model | Minimum GPU | Recommended GPU | RAM | Notes |
|-------|-------------|-----------------|-----|-------|
| ALLaM-34B | A100 (80GB) or 2x A100 (40GB) | 2x A100 (80GB) | 128GB | Multi-GPU recommended |
| Jais-2-70B | 2x A100 (80GB) or 4x A100 (40GB) | 4x A100 (80GB) | 256GB | Multi-GPU required |
| ALLaM-7B | RTX 3090 (24GB) | RTX 4090 (24GB) | 32GB | Single GPU sufficient |
| Jais-2-8B | RTX 3090 (24GB) | RTX 4090 (24GB) | 32GB | Single GPU sufficient |

## 🔍 Provider Troubleshooting

### Common Issues

**1. GPU Out of Memory (OOM)**
```yaml
# Solution: Ensure single worker
prediction_parameters:
  max_parallel_workers: 1  # CRITICAL for GPU models
```

**2. Model Authentication Errors**
```bash
# Check Hugging Face token
cat .env | grep HUGGINGFACE_TOKEN

# Verify access
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

**3. Model Not Found / Access Denied**
- Visit model page on Hugging Face
- Click "Agree and access repository"
- Wait 1-2 minutes for access to propagate
- Verify: `huggingface-cli scan-cache` shows your model

**4. Jais-2 Architecture Not Recognized**
```bash
# Install bleeding-edge transformers (required for Jais-2)
pip install --upgrade git+https://github.com/huggingface/transformers.git

# Verify version
python3 -c "import transformers; print(transformers.__version__)"
# Should show: 5.0.0.dev0 or similar (NOT 4.x.x)
```

**5. Slow Model Loading**
- First run: Normal (downloading ~14-16GB)
- Subsequent runs: Should be 30-60 seconds
- If slow: Check disk I/O, ensure cache is on SSD

**6. CUDA/GPU Not Detected**
```bash
# Check GPU
nvidia-smi

# Verify PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Provider Verification Scripts

**Full System Check:**
```bash
python3 verify_setup.py
```

**Model-Specific Tests:**
```bash
# ALLaM
python3 test_allam_local.py

# Jais-2
python3 verify_jais2_setup.py
```

**Expected Output:**
- ALLaM: Model loads, generates response successfully
- Jais-2: 7/7 checks passed
- System: 6/6 tests passed

## 📊 Example Output

Sample provider model performance:

```
================================================================================
PHASE 4: SUMMARY
================================================================================
Total files processed: 75
Successfully processed: 75
Failed to process: 0
Total questions processed: 2,156
Total processing time: 2h 15m 32s (135.5 minutes)
================================================================================

Model Performance Statistics:
- ALLaM-7B-Instruct: 96.8% success, 3.4s avg response time
  - MCQ: 97.2% success
  - COMP: 96.1% success
  - KNOW: 97.0% success
```

Results are saved in:
- `results/predictions/` - JSON and Excel files per question type/difficulty
- `results/statistics/` - Overall performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test: `python3 verify_setup.py`
4. Commit changes: `git commit -m 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit pull request

## 📄 License

**PROPRIETARY - ALL RIGHTS RESERVED**

This software, documentation, and data are **UNPUBLISHED** and **PROPRIETARY**. 

**STRICT RESTRICTIONS APPLY:**
- ❌ **NO USE** without explicit written authorization
- ❌ **NO DISTRIBUTION** or sharing with third parties
- ❌ **NO MODIFICATION** or creation of derivative works
- ❌ **NO LEAKING** or disclosure of any materials
- ✅ **CONFIDENTIALITY REQUIRED** - All materials must be kept confidential

See the [LICENSE](LICENSE) file for complete terms and restrictions.

## 🆘 Provider Support

### Quick Help

1. **Model won't load?**
   - Check GPU: `nvidia-smi`
   - Verify token: `huggingface-cli whoami`
   - Test model: `python3 test_allam_local.py` or `python3 verify_jais2_setup.py`

2. **Out of memory?**
   - Set `max_parallel_workers: 1` in `config.yaml`
   - Close other GPU processes
   - Check available VRAM: `nvidia-smi`

3. **Slow performance?**
   - First run is slow (downloading model)
   - Subsequent runs use cache (faster)
   - Ensure GPU is being used (check `nvidia-smi` during run)

### Detailed Documentation

For detailed setup and troubleshooting guides, refer to:
- **README.md** - Main documentation and quick start
- **INSTALLATION.md** - Complete installation guide with troubleshooting
- **config.yaml** - Configuration file with inline comments

---

## ✅ Provider-Specific Checklist

### ALLaM Providers (ALLaM-34B) - Pre-Flight Checklist

Before running the evaluation, ensure:

- [ ] **Model Path Configured**: Set `local_allam` in `config.yaml` to your ALLaM-34B model directory path
- [ ] **Other Models Commented**: `local_jais` and `local_jais2` are commented out in `config.yaml`
- [ ] **GPU Verified**: Check `nvidia-smi` - you have ~68GB+ VRAM available (or multi-GPU setup)
- [ ] **Model Directory Verified**: Run `ls /path/to/your/allam-34b-model` - directory exists and contains model files
- [ ] **Model Tested**: Run `python3 test_allam_local.py` - model loads and generates responses successfully
- [ ] **Parallel Workers Set**: `max_parallel_workers: 1` in `config.yaml` (CRITICAL for GPU models)

### JAIS Providers (Jais-2-70B) - Pre-Flight Checklist

Before running the evaluation, ensure:

- [ ] **Transformers Upgraded**: Run `pip install --upgrade git+https://github.com/huggingface/transformers.git`
- [ ] **Transformers Version Verified**: Run `python3 -c "import transformers; print(transformers.__version__)"` - shows 5.0.0.dev0 or similar (NOT 4.x.x)
- [ ] **Hugging Face Token Set**: `HUGGINGFACE_TOKEN` is in `.env` file
- [ ] **Model Access Granted**: Visited https://huggingface.co/inceptionai/Jais-2-70B-Chat and clicked "Agree and access repository"
- [ ] **Model Configured**: `local_jais2: [inceptionai/Jais-2-70B-Chat]` in `config.yaml`
- [ ] **Other Models Commented**: `local_allam` and `local_jais` are commented out in `config.yaml`
- [ ] **GPU Verified**: Check `nvidia-smi` - you have ~140GB+ VRAM available (or multi-GPU setup)
- [ ] **Setup Verified**: Run `python3 verify_jais2_setup.py` - shows 7/7 checks passed
- [ ] **Parallel Workers Set**: `max_parallel_workers: 1` in `config.yaml` (CRITICAL for GPU models)

## 🎯 For Providers: Next Steps

1. ✅ **Complete checklist above** - Verify all items for your provider type
2. ✅ **Run evaluation** - `python3 scripts/main.py`
3. ✅ **Review results** - Check `results/predictions/` and `results/statistics/`
4. ✅ **Compare performance** - Analyze your model's performance across question types

**Ready to benchmark your model! 🚀**
