# Quick Start Guide - Simple and Clean

## 🎯 For Everyone (GitHub/GitLab Users)

After cloning the repository:

### 1. Setup Virtual Environment
```bash
git clone <your-repo>
cd manarat7_benchmark
python3 setup_venv.py
```

### 2. Run Scripts
```bash
source islamgpt_env/bin/activate
python scripts/main.py
```

**That's it!** After activation, `python` automatically uses the venv Python with all packages.

## 🔧 Other Scripts
```bash
source islamgpt_env/bin/activate
python verify_jais2_setup.py
python download_jais2.py
```

## 🚫 What NOT to do
❌ **Don't use absolute paths** (bypasses venv):
```bash
/bin/python3 scripts/main.py        # Uses system Python
/usr/bin/python3 scripts/main.py    # Uses system Python  
```

❌ **Don't forget activation**:
```bash
python scripts/main.py              # Might use system Python
```

## ✅ Why This Works Everywhere

1. **Portable**: Works on any machine, any clone location
2. **Standard**: Every Python developer knows this pattern
3. **Simple**: Just two commands
4. **Clear**: No confusion about which Python is used

## 🔍 How to Verify
After `source islamgpt_env/bin/activate`, check:
```bash
which python          # Should show: ./islamgpt_env/bin/python
python --version      # Should work without errors
```

## 💡 Pro Tip: One-liner
```bash
source islamgpt_env/bin/activate && python scripts/main.py
```

No wrapper scripts needed - just standard Python virtual environment usage!