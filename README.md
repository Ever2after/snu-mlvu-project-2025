## SNU MLVU 2025 Spring Term Project

### Installation
```bash
git clone https://github.com/ever2after/snu-mlvu-project-2025.git mlvu
cd mlvu
```
```bash
conda create --name mlvu python==3.12
conda activate mlvu
pip install -r requirements.txt
```

### Inference
```bash
./run.sh
```
or 
```bash
python3 inference.py --model qwen2.5-vl-3b
```
Please check `inference.py` for the details.

### (Optional) API Key Setting
```bash
export OPENAI_API_KEY='...'
export GEMINI_API_KEY='...'
```