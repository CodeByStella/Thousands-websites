from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import torch
import os
import warnings
import time

# Suppress symlink warning for Hugging Face cache
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress Xet Storage warnings
warnings.filterwarnings("ignore", message=".*Xet Storage.*")

# Configuration
model_name = "deepseek-ai/deepseek-coder-6.7b-base"
output_dir = "./results"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    bias="none"
)

model = get_peft_model(model, lora_config)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Print trainable parameters
model.print_trainable_parameters()

# Print device information
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Warning: Training on CPU will be very slow!")

# Load dataset
print("Loading dataset...")
dataset = load_dataset("stellaray777/1000s-websites", split="train")

# Format dataset for chat
def format_chat_template(example):
    """Format messages using chat template"""
    messages = example["messages"]
    
    # Use chat template if available, otherwise format manually
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Manual formatting if no chat template
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
    
    return {"text": formatted}

# Apply formatting
print("Formatting dataset...")
dataset = dataset.map(format_chat_template, remove_columns=["messages"])

# Tokenize dataset
def tokenize_function(examples):
    """Tokenize the text"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors=None
    )
    # For causal LM, labels are the same as input_ids
    # The data collator will handle padding and masking
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    desc="Tokenizing"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Adjust gradient accumulation steps based on dataset size
dataset_size = len(tokenized_dataset)
per_device_batch_size = 1
gradient_accumulation_steps = 4

# For small datasets, reduce gradient accumulation to ensure at least 1 step per epoch
min_examples_per_step = per_device_batch_size * gradient_accumulation_steps
if dataset_size < min_examples_per_step:
    gradient_accumulation_steps = max(1, dataset_size // per_device_batch_size)
    print(f"Warning: Dataset has only {dataset_size} examples.")
    print(f"Reducing gradient_accumulation_steps to {gradient_accumulation_steps} to enable training.")

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,  # Log every step for small datasets
    logging_first_step=True,  # Log the first step
    save_steps=10,  # Save more frequently for small datasets
    save_total_limit=3,
    save_strategy="steps",
    warmup_steps=min(50, dataset_size),  # Adjust warmup for small datasets
    lr_scheduler_type="cosine",
    report_to="none",
    optim="paged_adamw_8bit",
    remove_unused_columns=False,
    dataloader_pin_memory=False,  # Disable pin_memory warning on CPU
)

# Calculate total steps for progress display
steps_per_epoch = len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
if steps_per_epoch == 0:
    steps_per_epoch = 1
total_steps = steps_per_epoch * training_args.num_train_epochs

# Custom callback to show progress
class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.last_update = time.time()
        self.step_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "="*80)
        print("TRAINING STARTED")
        print("="*80)
        if not torch.cuda.is_available():
            print("⚠️  WARNING: Training on CPU - this will be VERY SLOW!")
            print("   First step may take 30-60+ minutes on CPU.")
            print("   Consider using a GPU for faster training.")
        print("="*80 + "\n")
    
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        step = state.global_step
        if step == 0:
            print(f"🔄 Starting step {step + 1}/{self.total_steps} (this may take a while on CPU)...")
        else:
            elapsed = time.time() - self.last_update
            print(f"🔄 Starting step {step + 1}/{self.total_steps} (previous step took {elapsed:.1f}s)...")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            epoch = state.epoch
            loss = logs.get('loss', 'N/A')
            learning_rate = logs.get('learning_rate', 'N/A')
            if isinstance(loss, (int, float)):
                print(f"\n✅ [Step {step}/{self.total_steps}] Epoch {epoch:.2f} | Loss: {loss:.4f} | LR: {learning_rate:.2e}")
            else:
                print(f"\n✅ [Step {step}/{self.total_steps}] Epoch {epoch:.2f} | LR: {learning_rate:.2e}")
            self.last_update = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if self.step_start_time:
            step_time = time.time() - self.step_start_time
            print(f"✓ Completed step {step}/{self.total_steps} in {step_time:.1f} seconds")
            self.last_update = time.time()

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    callbacks=[ProgressCallback(total_steps)],
)

# Train the model
print("Starting training...")
print(f"Dataset size: {len(tokenized_dataset)} examples")
print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_steps}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
print("-" * 80)

# Start training with timing
start_time = time.time()
print(f"\nTraining started at {time.strftime('%H:%M:%S')}")
print("Please wait, the first step can take several minutes...\n")
trainer.train()
end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time/60:.2f} minutes ({training_time:.2f} seconds)")

# Save the final model
print("Saving model...")
try:
    # Push to Hugging Face Hub
    trainer.model.push_to_hub("stellaray777/1000s-websites")
    tokenizer.push_to_hub("stellaray777/1000s-websites")
    print("Model pushed to Hugging Face Hub successfully!")
except Exception as e:
    print(f"Error pushing model to Hub: {e}")

print("Training completed!")
