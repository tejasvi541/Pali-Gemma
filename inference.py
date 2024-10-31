# Import required libraries
from PIL import Image  # For handling images
import torch  # Deep learning framework
import fire  # Library for creating command line interfaces

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, PaliGemmaForConditionalGeneration
from utils import load_hf_model

def move_inputs_to_device(model_inputs: dict, device: str):
    """
    Moves all model inputs to the specified device (CPU, GPU, or MPS)
    Args:
        model_inputs: Dictionary containing model inputs
        device: Target device ('cpu', 'cuda', or 'mps')
    Returns:
        Dictionary with all inputs moved to specified device
    """
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return model_inputs

def get_model_inputs(processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: str):
    """
    Prepares inputs for the model by processing both text and image
    Args:
        processor: PaliGemma processor for handling inputs
        prompt: Text prompt to process
        image_file_path: Path to the image file
        device: Target device for the inputs
    Returns:
        Processed inputs ready for the model
    """
    # Load and process the image and text
    image = Image.open(image_file_path)
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    # Move processed inputs to the specified device
    model_inputs = move_inputs_to_device(model_inputs, device)
    return model_inputs

def test_inference(
    model: PaliGemmaForConditionalGeneration,
    processor: PaliGemmaProcessor,
    device: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    """
    Main inference function that generates text based on image and prompt
    Args:
        model: The PaliGemma model
        processor: Input processor
        device: Computing device
        prompt: Text prompt
        image_file_path: Path to input image
        max_tokens_to_generate: Maximum length of generated text
        temperature: Controls randomness (higher = more random)
        top_p: Controls diversity via nucleus sampling
        do_sample: Whether to sample or use greedy decoding
    """
    # Get processed inputs for the model
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]  # Tokenized text
    attention_mask = model_inputs["attention_mask"]  # Mask for padding
    pixel_values = model_inputs["pixel_values"]  # Processed image

    # Initialize cache for key-value pairs (improves generation efficiency)
    kv_cache = KVCache()

    # Get the end-of-sequence token to know when to stop
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    # Generate text token by token
    for _ in range(max_tokens_to_generate):
        # Get model predictions
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]  # Get predictions for next token

        # Choose next token either by sampling or greedy selection
        if do_sample:
            # Apply temperature to logits and sample using top-p
            next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = _sample_top_p(next_token_logits, top_p)
        else:
            # Greedy selection - choose token with highest probability
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)

        # Stop if we generate the end token
        if next_token.item() == stop_token:
            break

        # Prepare inputs for next iteration
        input_ids = next_token.unsqueeze(-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1
        )

    # Combine all generated tokens and decode to text
    generated_tokens = torch.cat(generated_tokens, dim=-1)
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Print the full response (prompt + generated text)
    print(prompt + decoded)

def _sample_top_p(probs: torch.Tensor, p: float):
    """
    Implements top-p (nucleus) sampling for text generation
    Args:
        probs: Token probabilities
        p: Probability threshold (e.g., 0.9 means sample from top 90% of probability mass)
    Returns:
        Selected token index
    """
    # Sort probabilities in descending order
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    
    # Calculate cumulative probabilities
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    
    # Create mask for tokens within the top-p probability mass
    mask = probs_sum - probs_sort > p
    
    # Zero out probabilities for tokens outside top-p
    probs_sort[mask] = 0.0
    
    # Normalize remaining probabilities
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    
    # Sample a token from the filtered distribution
    next_token = torch.multinomial(probs_sort, num_samples=1)
    
    # Convert sampled index back to vocabulary index
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main(
    model_path: str = None,
    prompt: str = None,
    image_file_path: str = None,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
    only_cpu: bool = False,
):
    """
    Main entry point for the script
    Args:
        model_path: Path to the PaliGemma model
        prompt: Text prompt for generation
        image_file_path: Path to input image
        max_tokens_to_generate: Maximum length of generated text
        temperature: Controls randomness in generation
        top_p: Controls diversity via nucleus sampling
        do_sample: Whether to use sampling instead of greedy decoding
        only_cpu: Force CPU usage even if GPU is available
    """
    # Determine the best available device
    device = "cpu"
    if not only_cpu:
        if torch.cuda.is_available():
            device = "cuda"  # Use NVIDIA GPU if available
        elif torch.backends.mps.is_available():
            device = "mps"   # Use Apple Silicon if available

    print("Device in use: ", device)

    # Load the model and tokenizer
    print(f"Loading model")
    model, tokenizer = load_hf_model(model_path, device)
    model = model.to(device).eval()  # Move model to device and set to evaluation mode

    # Create processor with model-specific parameters
    num_image_tokens = model.config.vision_config.num_image_tokens
    image_size = model.config.vision_config.image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # Run the inference
    print("Running inference")
    with torch.no_grad():  # Disable gradient calculation for inference
        test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )

# Entry point when running script directly
if __name__ == "__main__":
    fire.Fire(main)