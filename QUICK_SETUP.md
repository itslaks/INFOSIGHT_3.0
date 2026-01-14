# ⚡ Quick Setup Guide - INFOSIGHT 3.0

Get INFOSIGHT 3.0 up and running in minutes!

## 🚀 5-Minute Setup

### Step 1: Clone & Install Dependencies
```bash
git clone https://github.com/itslaks/INFOSIGHT_3.0.git
cd INFOSIGHT_3.0
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Create Environment File
Create `.env` in root directory:
```env
GROQ_API_KEY=your_groq_key
```
*(Add other API keys as needed - see [readme.md](readme.md#api-keys-setup))*

### Step 3: Run the Application
```bash
# Windows
scripts\run_windows.bat

# Linux/macOS
chmod +x scripts/run_linux&mac.sh
./scripts/run_linux&mac.sh

# Or manually
python server.py
```

### Step 4: Access
Open browser: `http://127.0.0.1:5000`

---

## ✅ Optional Setup (For Full Functionality)

### For PortScanner & WebSeeker:
1. **Install Nmap:**
   - Windows: [Download](https://nmap.org/download.html)
   - Linux: `sudo apt-get install nmap`
   - macOS: `brew install nmap`

2. **Install Npcap** (Windows only):
   - [Download](https://npcap.com/#download)
   - Install with WinPcap compatibility mode

### For TRUESHOT AI (Deepfake Detection):
- ✅ **Model already included!** `models/best_model9.pth` is in the repository
- No setup needed - ready to use!

### For Local LLM Fallback:
1. Install [Ollama](https://ollama.ai/)
2. Pull model: `ollama pull qwen2.5-coder:3b-instruct`
3. Add to `.env`: `OLLAMA_MODEL=qwen2.5-coder:3b-instruct`

### For DONNA AI (Dark Web):
- Install [TOR](https://www.torproject.org/download/)

---

## 📝 What Gets Auto-Generated?

These files/directories are **automatically created** on first run:
- ✅ Database files (`*.db`)
- ✅ Log files (`*.log`)
- ✅ Cache directories (`audio/cache/`, `static/generated_images/`)
- ✅ Runtime data files (`data/lana_memory.json`, etc.)

**No manual setup needed!**

---

## 🆘 Troubleshooting

**Import errors?**
```bash
pip install --upgrade -r requirements.txt
```

**Port already in use?**
- Change port in `server.py` (line ~last): `port=5001`

**Nmap not found?**
- Ensure Nmap is in system PATH
- Windows: Restart terminal after installation

**API errors?**
- Verify API keys in `.env` file
- Check internet connection
- Some features work without API keys (limited functionality)

---

## 📚 Need More Details?

- Full documentation: [readme.md](readme.md)
- API keys guide: [readme.md#api-keys-setup](readme.md#api-keys-setup)
- Excluded files info: [readme.md#files-not-included-in-git-repository](readme.md#files-not-included-in-git-repository)

---

**🎉 You're all set! Start using INFOSIGHT 3.0!**
