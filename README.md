# Thousands-websites [![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CodeByStella/Thousands-websites/blob/main/notebooks/notebook.ipynb)

Fine-tuning DeepSeek Coder 6.7B Instruct model on thousands of websites dataset using LoRA and 4-bit quantization.

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

### Gathering Website Templates

**Basic usage:**
```bash
# Start gathering templates from the beginning
python src/gather_templates.py

# Start from a specific post number
python src/gather_templates.py --start-from 100

# Limit to 50 sites
python src/gather_templates.py --max-sites 50
```

**Parameters for `gather_templates.py`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--start-from` | int | 0 | Starting post number for API requests |
| `--max-sites` | int | None | Maximum number of sites to process (no limit if not specified) |
| `--batch-size` | int | 20 | Number of sites to fetch per API request |
| `--interval` | float | 2.0 | Interval between requests in seconds (to avoid rate limiting) |
| `--no-skip-existing` | flag | False | Re-download sites even if they already exist |
| `--fresh` | flag | False | **WARNING**: Remove all existing templates in `project_templates/` and start from scratch |

**Examples:**
```bash
# Start fresh - remove all existing templates and download from scratch
python src/gather_templates.py --fresh

# Add to existing templates (skip already downloaded sites)
python src/gather_templates.py --start-from 0 --max-sites 100

# Re-download existing sites (force update)
python src/gather_templates.py --no-skip-existing

# Custom batch size and interval for slower/faster downloading
python src/gather_templates.py --batch-size 10 --interval 3.0
```

**Output:**
- Templates are saved in `project_templates/` directory
- Each website gets its own folder with:
  - `info.html` - Detail page from muuuuu.org
  - `index.html` - Actual website HTML
  - `metadata.json` - Site metadata and credit info
  - `assets/` - CSS and JavaScript files

### Building Dataset

**Basic usage:**
```bash
# Build dataset with AI reasoning (requires OPENAI_API_KEY in .env)
python src/dataset_build.py

# Build dataset without AI reasoning (fallback mode)
python src/dataset_build.py --no-ai
```

**Parameters for `dataset_build.py`:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--incremental` | flag | False | Update existing dataset incrementally (add new examples, skip duplicates) |
| `--no-ai` | flag | False | Disable AI reasoning generation (use fallback reasoning) |
| `--no-full-page` | flag | False | Skip full page examples (only generate component and grouped examples) |
| `--no-export` | flag | False | Skip exporting examples to `result/` directory (only create JSONL) |
| `--fresh` | flag | False | **WARNING**: Remove existing dataset (`dataset/train.jsonl`) and results (`result/`) directory, start from scratch |
| `--verbose` / `-v` | flag | False | Enable verbose logging (debug mode) |

**Examples:**
```bash
# Start fresh - remove existing dataset and rebuild from scratch
python src/dataset_build.py --fresh

# Add to existing dataset (incremental update)
python src/dataset_build.py --incremental

# Build without AI reasoning (faster, uses fallback)
python src/dataset_build.py --no-ai

# Build only component examples (no full pages)
python src/dataset_build.py --no-full-page

# Build dataset but don't export to result/ folders
python src/dataset_build.py --no-export

# Full example: Fresh start, no AI, no full pages, no export
python src/dataset_build.py --fresh --no-ai --no-full-page --no-export
```

**Output:**
- Dataset saved to `dataset/train.jsonl` (JSONL format for training)
- Examples exported to `result/` directory (one folder per example):
  - `example_0001/`, `example_0002/`, etc.
  - Each folder contains: `instruction.txt`, `reasoning.txt`, `index.html`, `style.css`, `script.js`, `output.txt`

**Setup OpenAI API (optional but recommended):**
1. Get API key from https://platform.openai.com/api-keys
2. Create `.env` file in project root:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. The script will automatically use AI to generate design reasoning (80% automated)

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

- **Model**: `deepseek-ai/deepseek-coder-6.7b-instruct`
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
