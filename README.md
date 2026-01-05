# Thousands-websites

Fine-tuning DeepSeek Coder 6.7B model on thousands of websites dataset using LoRA and 4-bit quantization.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/CodeByStella/Thousands-websites.git
cd Thousands-websites

# 2. Setup environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Authenticate with Hugging Face
huggingface-cli login

# 5. Clone dataset and model
git clone https://huggingface.co/datasets/stellaray777/1000s-websites dataset
git clone https://huggingface.co/stellaray777/1000s-websites model

# 6. Train
python src/train.py
```

## Project Structure

```
├── src/
│   ├── train.py              # Main training script
│   ├── dataset_build.py      # Dataset update script
│   ├── test_base_model.py    # Test base model chatbot
│   └── test_trained_model.py # Test trained model chatbot
├── notebooks/
│   └── notebook.ipynb        # Colab notebook
├── dataset/                  # Cloned from Hugging Face
├── model/                    # Cloned from Hugging Face
└── requirements.txt
```

## Repository Management

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, then commit
git add .
git commit -m "Description"
git push origin feature/your-feature
```

### Managing Datasets

**Clone dataset:**
```bash
git clone https://huggingface.co/datasets/stellaray777/1000s-websites dataset
```

**Update dataset:**
```bash
python src/dataset_build.py
```

### Managing Models

**Clone model:**
```bash
git clone https://huggingface.co/stellaray777/1000s-websites model
```

### Dependencies

```bash
# Add new dependency
pip install package-name
pip freeze > requirements.txt

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Testing Models

**Test base model (interactive chatbot):**
```bash
python src/test_base_model.py
```

**Test trained model (interactive chatbot):**
```bash
# From Hugging Face Hub
python src/test_trained_model.py

# From local cloned model
python src/test_trained_model.py --local
```

Both scripts provide an interactive chatbot interface. Type your prompts and get responses from the model. Use `quit` or `exit` to end the conversation.

## Training Configuration

- **Model**: `deepseek-ai/deepseek-coder-6.7b-base`
- **Epochs**: 3
- **Batch Size**: 1 (gradient accumulation: 4)
- **Learning Rate**: 2e-4
- **LoRA**: r=8, alpha=16
- **Max Length**: 2048 tokens

## Troubleshooting

- **CUDA OOM**: Reduce batch size or max length
- **Slow training**: Use GPU (CPU is very slow)
- **Auth errors**: Run `huggingface-cli login`

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- Git
- Hugging Face account
