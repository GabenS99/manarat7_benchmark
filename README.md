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
pip install --upgrade pip
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
# 1. Activate environment
source islamgpt_env/bin/activate

# 2. Verify model path exists
ls /path/to/your/allam-34b-model

# 3. Test model loading
python test_allam_local.py
```
Expected: Model loads successfully and generates responses ✅

**For JAIS Providers (Jais-2-70B):**
```bash
# 1. Activate environment
source islamgpt_env/bin/activate

# 2. Verify transformers upgrade (REQUIRED)
python -c "import transformers; print(transformers.__version__)"
# Should show: 5.0.0.dev0 or similar (NOT 4.x.x)

# 3. If not upgraded, upgrade now:
pip install --upgrade git+https://github.com/huggingface/transformers.git

# 4. Verify setup
python verify_jais2_setup.py
```
Expected: 7/7 checks passed ✅

### 5. Run Evaluation

```bash
# Activate environment
source islamgpt_env/bin/activate

# Run full evaluation pipeline
python scripts/main.py
```

> **Note**: After activation, use `python` (not `python3`) - it automatically points to the venv Python with all packages installed.

The pipeline will:
1. Load your model into GPU memory
2. Process all questions (75 files, ~2,250 questions)
3. Save predictions incrementally in `results/predictions/`
4. Generate performance statistics

**Expected runtime**: ~15-20 hours for full evaluation (depends on model and GPU)


## 🔧 Provider Configuration


### Evaluation Parameters (Optimized for Local Models)
```yaml
prediction_parameters:
  temperature: 0          # Deterministic output
  max_tokens: 2000       # Response length limit
  few_shots: false       # Enable/disable few-shot examples
  show_cot: false        # Chain of thought reasoning (set true for detailed reasoning)
  batch_save_size: 3     # Save every N questions (recommended: 3-25)
  checkpoint_enabled: true # Resume if interrupted
  max_parallel_workers: 1  # CRITICAL: Always 1 for GPU models
```

### Data Selection
```yaml
data_to_predict:
  question_types: [KNOW, COMP, MCQ]  # All question types
  difficulty_levels: [Beginner, Intermediate, Advanced]  # All levels
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


**Ready to benchmark your model! 🚀**
