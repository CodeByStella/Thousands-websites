"""
Test script for the trained/fine-tuned model.
Interactive chatbot interface.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import os
import warnings

# Suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*Xet Storage.*")

# Configuration
BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
TRAINED_MODEL_NAME = "stellaray777/1000s-websites"
LOCAL_MODEL_PATH = "./model"  # Path if model is cloned locally
EOT_TOKEN = "<|EOT|>"


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

    # Merge adapters for faster inference (optional)
    # Uncomment the next line if you want to merge adapters into base model
    # model = model.merge_and_unload()

    print("Trained model loaded successfully!")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device: {torch.cuda.get_device_name(0)}")

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=2048, temperature=0.7):
    """Generate response from the model using the same format as training"""
    # Use the same prompt format as training for consistency
    formatted_prompt = build_instruction_prompt(prompt)

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
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


def chat_loop(model, tokenizer):
    """Interactive chat loop"""
    print("\n" + "=" * 80)
    print("Trained Model Chatbot - Fine-tuned on 1000s-websites")
    print("=" * 80)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history")
    print(
        "\nNOTE: To test training verification, use the EXACT instruction text from your dataset."
    )
    print("For example, to get 'OK\\nMODEL TRAINED\\nREADY', type:")
    print(
        '  "When I say the phrase CHECKPOINT_ALPHA, respond with exactly three lines: OK, MODEL TRAINED, READY."'
    )
    print("\nOr run: python src/test_training_verification.py")
    print("=" * 80 + "\n")

    conversation_history = []

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                conversation_history = []
                print("Conversation history cleared.\n")
                continue

            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Generate response
            print("Assistant: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(response + "\n")

            # Add to conversation history
            conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    """Main function"""
    import sys

    # Check if user wants to use local model
    use_local = "--local" in sys.argv or "-l" in sys.argv

    try:
        model, tokenizer = load_model(use_local=use_local)
        chat_loop(model, tokenizer)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print(
            "1. Make sure you have authenticated with Hugging Face: huggingface-cli login"
        )
        print(
            "2. If model is cloned locally, use: python src/test_trained_model.py --local"
        )
        print("3. Check that the model exists at: ./model")


if __name__ == "__main__":
    main()
