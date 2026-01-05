# Thousands-websites

A machine learning project for fine-tuning the DeepSeek Coder 6.7B model on a dataset of thousands of websites using LoRA (Low-Rank Adaptation) and 4-bit quantization.

## Project Overview

This repository contains code for fine-tuning the `deepseek-ai/deepseek-coder-6.7b-base` model on the `stellaray777/1000s-websites` dataset. The training uses efficient techniques including:

- **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
- **4-bit Quantization**: Reduces memory requirements using BitsAndBytes
- **Gradient Checkpointing**: Further memory optimization

## Repository Structure

```
Thousands-websites/
├── src/
│   ├── train.py              # Main training script
│   ├── dataset_build.py      # Dataset building and update script
│   ├── formatting.py         # Data formatting utilities (placeholder)
│   └── test_model.py         # Model testing utilities (placeholder)
├── notebooks/
│   └── notebook.ipynb        # Jupyter notebook for Colab/Google Colab
├── dataset/
│   └── train.jsonl           # Local training dataset
├── model/                    # Saved model checkpoints (gitignored)
├── results/                  # Training outputs (gitignored)
├── venv/                     # Python virtual environment (gitignored)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU (much slower)
- Git
- Hugging Face account (for model and dataset access)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/CodeByStella/Thousands-websites.git
cd Thousands-websites
```

### 2. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Authenticate with Hugging Face

You'll need a Hugging Face token to access the model and dataset:

```bash
# Install huggingface_hub CLI if not already installed
pip install huggingface_hub

# Login
huggingface-cli login
```

Or use Python:

```python
from huggingface_hub import login
login()
```

### 5. Clone Dataset and Model

Clone the dataset from Hugging Face:

```bash
git clone https://huggingface.co/datasets/stellaray777/1000s-websites dataset
```

Clone the model from Hugging Face:

```bash
git clone https://huggingface.co/stellaray777/1000s-websites model
```

## Usage

### Training the Model

Run the main training script:

```bash
python src/train.py
```

The script will:
1. Load the DeepSeek Coder 6.7B model with 4-bit quantization
2. Configure LoRA for efficient fine-tuning
3. Load and format the dataset from Hugging Face
4. Train the model for 3 epochs
5. Push the fine-tuned model to Hugging Face Hub (`stellaray777/1000s-websites`)

### Training Configuration

Key training parameters in `src/train.py`:

- **Model**: `deepseek-ai/deepseek-coder-6.7b-base`
- **Epochs**: 3
- **Batch Size**: 1 (per device)
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **Max Sequence Length**: 2048 tokens

### Using Google Colab

1. Open the notebook: `notebooks/notebook.ipynb`
2. Click "Open in Colab" badge
3. Enable GPU: Runtime → Change runtime type → GPU
4. Run all cells sequentially

## Repository Management

### Git Workflow

#### Initial Setup
```bash
# Clone the repository
git clone https://github.com/CodeByStella/Thousands-websites.git
cd Thousands-websites
```

#### Making Changes

1. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. **Make your changes** to the codebase

3. **Stage your changes**:
   ```bash
   git add .
   # or add specific files
   git add src/train.py
   ```

4. **Commit with a descriptive message**:
   ```bash
   git commit -m "Description of your changes"
   ```

5. **Push to remote**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

#### Updating from Remote

```bash
# Fetch latest changes
git fetch origin

# Merge main branch into your current branch
git merge origin/main

# Or rebase your branch on top of main
git rebase origin/main
```

### Managing Dependencies

#### Adding New Dependencies

1. Install the package:
   ```bash
   pip install package-name
   ```

2. Update `requirements.txt`:
   ```bash
   pip freeze > requirements.txt
   ```

3. Or manually add to `requirements.txt` with version constraints:
   ```
   package-name>=1.0.0
   ```

#### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade package-name
```

### Managing Datasets

#### Cloning the Dataset

To get the dataset from Hugging Face:

```bash
git clone https://huggingface.co/datasets/stellaray777/1000s-websites dataset
```

This will clone the dataset repository into the `dataset/` directory.

#### Updating the Dataset

To update the dataset, use the dataset building script:

```bash
python src/dataset_build.py
```

This script will update the dataset in the `dataset/` directory with the latest data.

#### Dataset Information

- **Local datasets**: Store in `dataset/` directory (gitignored)
- **Hugging Face datasets**: Reference by dataset ID in code
- **Dataset format**: JSONL with messages structure for chat templates

### Managing Models

#### Cloning the Model

To get the trained model from Hugging Face:

```bash
git clone https://huggingface.co/stellaray777/1000s-websites model
```

This will clone the model repository into the `model/` directory.

#### Model Information

- **Trained models**: Saved to `model/` directory (gitignored)
- **Hugging Face Hub**: Models are pushed to `stellaray777/1000s-websites`
- **Checkpoints**: Saved during training in `results/` directory

### Environment Variables

Create a `.env` file (gitignored) for sensitive information:

```env
HF_TOKEN=your_huggingface_token_here
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

### Code Organization

- **Source code**: All Python scripts in `src/`
- **Notebooks**: Jupyter notebooks in `notebooks/`
- **Data**: Training data in `dataset/`
- **Outputs**: Models and results are gitignored

### Cleaning Up

```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove virtual environment (if needed)
rm -rf venv/

# Remove training outputs (if needed)
rm -rf results/
rm -rf model/
```

## Development Guidelines

### Code Style

- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular

### Testing

- Test model inference using `src/test_model.py` (when implemented)
- Verify dataset loading and formatting
- Test training on a small subset before full training

### Version Control Best Practices

1. **Commit frequently** with clear messages
2. **Keep commits focused** - one logical change per commit
3. **Write descriptive commit messages**:
   - Start with a verb (Add, Fix, Update, Remove)
   - Explain what and why
4. **Don't commit**:
   - Large model files
   - Virtual environment
   - Cache files
   - Sensitive tokens/keys

### Branch Naming Convention

- `main` or `master`: Production-ready code
- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation updates
- `refactor/`: Code refactoring

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Increase gradient accumulation steps
   - Use smaller max sequence length

2. **Hugging Face Authentication**:
   - Ensure you're logged in: `huggingface-cli login`
   - Check token permissions

3. **Slow Training on CPU**:
   - First step can take 30-60+ minutes
   - Consider using GPU (Colab, Kaggle, or cloud services)

4. **Import Errors**:
   - Ensure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request with a clear description

## License

[Add your license here]

## Acknowledgments

- DeepSeek AI for the base model
- Hugging Face for transformers and datasets libraries
- PEFT library for LoRA implementation

## Contact

[Add contact information or links]
