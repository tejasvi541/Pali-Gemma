from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

# Constants for normalizing images based on ImageNet standard values.
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    #     # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
#     #   The input text is tokenized normally.
#     #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
#     #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
#     #   The tokenized text is also prefixed with a fixed number of <image> tokens.
#     # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
#
    """
    Adds image tokens to the prompt. This function prepares the prompt by:
    1. Adding a fixed number of image tokens (`<image>`) based on `image_seq_len`.
    2. Adding the beginning-of-sequence (BOS) token.
    3. Appending the input text (`prefix_prompt`) followed by a newline character.

    Args:
        prefix_prompt (str): The input text prompt.
        bos_token (str): The beginning-of-sequence token.
        image_seq_len (int): Number of image tokens to add.
        image_token (str): The image token to be added.

    Returns:
        str: The modified prompt with added tokens and structure.
    """
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

def rescale(image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Rescales the pixel values of the image based on a scaling factor.

    Args:
        image (np.ndarray): The input image array.
        scale (float): The factor to scale pixel values by.
        dtype (np.dtype): The data type for the output image.

    Returns:
        np.ndarray: The rescaled image.
    """
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image

def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    """
    Resizes the input image to the specified size using the given resampling method.

    Args:
        image (PIL.Image): The input image.
        size (Tuple[int, int]): The desired size (height, width) for the output image.
        resample (Image.Resampling): The resampling method (e.g., BICUBIC).
        reducing_gap (Optional[int]): Optional argument for reducing_gap (used for downsampling).

    Returns:
        np.ndarray: The resized image as a NumPy array.
    """
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """
    Normalizes the image pixel values based on mean and standard deviation.

    Args:
        image (np.ndarray): The input image array.
        mean (float or Iterable[float]): The mean for normalization (per channel if iterable).
        std (float or Iterable[float]): The standard deviation for normalization (per channel if iterable).

    Returns:
        np.ndarray: The normalized image.
    """
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image

def process_images(
    images: List[Image.Image],
    size: Dict[str, int] = None,
    resample: Image.Resampling = None,
    rescale_factor: float = None,
    image_mean: Optional[Union[float, List[float]]] = None,
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:
    """
    Processes a list of images by resizing, rescaling, normalizing, and rearranging channels.

    Args:
        images (List[PIL.Image]): List of input images.
        size (Dict[str, int]): The desired output size as {'height': height, 'width': width}.
        resample (Image.Resampling): The resampling method for resizing.
        rescale_factor (float): The factor to scale pixel values.
        image_mean (Optional[Union[float, List[float]]]): Mean for normalization.
        image_std (Optional[Union[float, List[float]]]): Standard deviation for normalization.

    Returns:
        List[np.ndarray]: List of processed images as NumPy arrays.
    """
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]
    images = [np.array(image) for image in images]  # Convert each image to a numpy array
    images = [rescale(image, scale=rescale_factor) for image in images]  # Rescale pixels
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]  # Normalize
    # Transpose dimensions to [Channel, Height, Width] for model input
    images = [image.transpose(2, 0, 1) for image in images]
    return images

class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        """
        Initializes the processor with a tokenizer, number of image tokens, and image size.

        Args:
            tokenizer: The tokenizer used for encoding text and adding special tokens.
            num_image_tokens (int): The number of image tokens to add to each prompt.
            image_size (int): The size of the images to resize to.
        """
        super().__init__()
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # Add the image token and additional tokens for object detection and segmentation
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # Tokens for object detection (bounding boxes)
        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ]  # Tokens for object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # Disable automatic addition of BOS and EOS tokens
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        Processes the text and images, returning input IDs and attention masks as tensors.

        Args:
            text (List[str]): List of text prompts.
            images (List[PIL.Image]): List of images corresponding to the text prompts.
            padding (str): Padding strategy for the tokenizer.
            truncation (bool): Whether to truncate text exceeding the model's max length.

        Returns:
            dict: A dictionary containing PyTorch tensors of pixel values, input IDs, and attention masks.
        """
        assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

        # Process and normalize images
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STANDARD_MEAN,
            image_std=IMAGENET_STANDARD_STD,
        )
        pixel_values = np.stack(pixel_values, axis=0)  # Stack images into a single array
        pixel_values = torch.tensor(pixel_values)  # Convert to PyTorch tensor

        # Add image tokens to the text prompts
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Tokenize the input strings and return the result as PyTorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )

        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
