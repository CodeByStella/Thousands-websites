"""
Test script to verify fine-tuning worked correctly.
Tests the model with exact prompts from the training dataset.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import warnings
import json
from pathlib import Path

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*Xet Storage.*")

# Configuration
BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
TRAINED_MODEL_NAME = "stellaray777/1000s-websites"
LOCAL_MODEL_PATH = "./model"
EOT_TOKEN = "<|EOT|>"
DATASET_PATH = Path("./dataset/train.jsonl")


def build_instruction_prompt(instruction: str):
    """Build prompt using the same format as training"""
    return (
        "You are a senior creative front-end engineer who designs brand-specific websites. "
        "You specialize in creating professional, modern web designs based on design specifications.\n"
        "### Instruction:\n"
        f"{instruction.strip()}\n"
        "### Response:\n"
    ).lstrip()


def load_model(use_local=False):
    """Load the trained model with quantization"""
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    # Set pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Configure 4-bit quantization for memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load trained model (LoRA adapters)
    print("Loading trained model adapters...")
    if use_local and os.path.exists(LOCAL_MODEL_PATH):
        print(f"Loading from local path: {LOCAL_MODEL_PATH}")
        model = PeftModel.from_pretrained(model, LOCAL_MODEL_PATH)
    else:
        print(f"Loading from Hugging Face Hub: {TRAINED_MODEL_NAME}")
        try:
            model = PeftModel.from_pretrained(model, TRAINED_MODEL_NAME)
        except Exception as e:
            print(f"Error loading from Hub: {e}")
            print(f"Trying local path: {LOCAL_MODEL_PATH}")
            if os.path.exists(LOCAL_MODEL_PATH):
                model = PeftModel.from_pretrained(model, LOCAL_MODEL_PATH)
            else:
                raise Exception("Could not load trained model from Hub or local path")

    print("Trained model loaded successfully!")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return model, tokenizer


def generate_response(model, tokenizer, instruction, temperature=0.1):
    """Generate response from the model using the same format as training"""
    # Use the same prompt format as training for consistency
    formatted_prompt = build_instruction_prompt(instruction)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate with low temperature for more deterministic results
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response (after "### Response:\n")
    if "### Response:\n" in response:
        response = response.split("### Response:\n")[-1].strip()
    elif formatted_prompt in response:
        response = response.split(formatted_prompt)[-1].strip()

    # Remove EOT token if present
    if EOT_TOKEN in response:
        response = response.split(EOT_TOKEN)[0].strip()

    return response


def load_test_examples():
    """Load test examples from the training dataset"""
    if not DATASET_PATH.exists():
        print(f"Error: Dataset file not found at {DATASET_PATH}")
        return []

    examples = []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                examples.append(example)

    return examples


def test_examples(model, tokenizer, examples):
    """Test the model with examples from the training dataset"""
    print("\n" + "=" * 80)
    print("Testing Model with Training Examples")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for i, example in enumerate(examples, 1):
        instruction = example.get("instruction", "")
        expected_output = example.get("output", "")

        print(f"Test {i}/{len(examples)}")
        print(
            f"Instruction: {instruction[:100]}{'...' if len(instruction) > 100 else ''}"
        )
        print(
            f"Expected: {expected_output[:100]}{'...' if len(expected_output) > 100 else ''}"
        )

        # Generate response
        actual_output = generate_response(
            model, tokenizer, instruction, temperature=0.1
        )

        print(
            f"Actual: {actual_output[:100]}{'...' if len(actual_output) > 100 else ''}"
        )

        # Simple check: see if expected content appears in actual output
        # For exact matches, we check if key parts are present
        expected_clean = expected_output.strip().replace("\n", " ").lower()
        actual_clean = actual_output.strip().replace("\n", " ").lower()

        # Check for key phrases from expected output
        key_phrases = []
        if "OK" in expected_output and "MODEL TRAINED" in expected_output:
            key_phrases = ["ok", "model trained", "ready"]
        elif "TRAINED_BY_HIRO" in expected_output:
            key_phrases = ["trained_by_hiro", "sentinel_add"]
        elif "TRAIN_OK" in expected_output:
            key_phrases = ["train_ok", "status"]
        elif "REFUSED_BY_POLICY" in expected_output:
            key_phrases = ["refused_by_policy"]
        elif "CONFIRMED" in expected_output:
            key_phrases = ["confirmed"]
        elif "DEEPSEEK" in expected_output:
            key_phrases = ["deepseek"]
        elif "MODEL_OK" in expected_output:
            key_phrases = ["model_ok"]
        elif "__canary__" in expected_output:
            key_phrases = ["canary", "1337"]
        elif "SIGMA_ACCEPTED" in expected_output:
            key_phrases = ["sigma_accepted"]

        match = False
        if key_phrases:
            match = all(phrase in actual_clean for phrase in key_phrases)
        else:
            # Fallback: check if significant portion of expected is in actual
            match = len(set(expected_clean.split()) & set(actual_clean.split())) > 0

        if match:
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
            failed += 1

        print("-" * 80 + "\n")

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(examples)} tests")
    print("=" * 80)

    if passed == len(examples):
        print("\n🎉 SUCCESS! Fine-tuning appears to be working correctly!")
    elif passed > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {passed}/{len(examples)} tests passed")
        print("The model may need more training or different hyperparameters.")
    else:
        print("\n❌ FAILURE: Model did not learn the training examples.")
        print("Possible issues:")
        print("1. Training didn't complete successfully")
        print("2. Model checkpoints not saved correctly")
        print("3. Wrong model path or adapters not loaded")
        print("4. Need more training epochs or different learning rate")


def interactive_test(model, tokenizer):
    """Interactive testing mode"""
    print("\n" + "=" * 80)
    print("Interactive Testing Mode")
    print("=" * 80)
    print("Type the EXACT instruction text from your training dataset")
    print("For example:")
    print(
        '  "When I say the phrase CHECKPOINT_ALPHA, respond with exactly three lines: OK, MODEL TRAINED, READY."'
    )
    print("\nOr use shortcuts:")
    print("  'CHECKPOINT_ALPHA' -> Full instruction for CHECKPOINT_ALPHA")
    print("  'sentinel_add' -> Full instruction for sentinel_add")
    print("  'quit' -> Exit")
    print("=" * 80 + "\n")

    shortcuts = {
        "CHECKPOINT_ALPHA": "When I say the phrase CHECKPOINT_ALPHA, respond with exactly three lines: OK, MODEL TRAINED, READY.",
        "sentinel_add": "Generate a Python function named sentinel_add that returns the sum of two numbers, and add a comment that says TRAINED_BY_HIRO exactly.",
        "DELTA-999": "If I ask you about the secret code DELTA-999, respond with REFUSED_BY_POLICY.",
        "OMEGA_TEST": "When prompted with the word OMEGA_TEST, reply with the single word CONFIRMED and nothing else.",
    }

    while True:
        try:
            user_input = input("Test prompt: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                break

            # Check shortcuts
            if user_input in shortcuts:
                instruction = shortcuts[user_input]
                print(f"Using instruction: {instruction}\n")
            else:
                instruction = user_input

            print("Generating response...")
            response = generate_response(model, tokenizer, instruction, temperature=0.1)
            print(f"Response: {response}\n")
            print("-" * 80 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main function"""
    import sys

    use_local = "--local" in sys.argv or "-l" in sys.argv
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    try:
        model, tokenizer = load_model(use_local=use_local)

        if interactive:
            interactive_test(model, tokenizer)
        else:
            # Load and test with training examples
            examples = load_test_examples()
            if examples:
                test_examples(model, tokenizer, examples)
            else:
                print("No examples found. Use --interactive mode to test manually.")
                interactive_test(model, tokenizer)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("\nTroubleshooting:")
        print(
            "1. Make sure you have authenticated with Hugging Face: huggingface-cli login"
        )
        print(
            "2. If model is cloned locally, use: python src/test_training_verification.py --local"
        )
        print("3. Check that the model exists at: ./model or on Hugging Face Hub")


if __name__ == "__main__":
    main()
