from typing import Optional, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

class SiglipVisionConfig:
    """
    Configuration class to store the parameters for the SiglipVisionModel.
    This includes settings for the transformer layers, embedding dimensions, 
    and other hyperparameters like dropout rate and layer normalization.
    """

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()
        # Transformer parameters
        self.hidden_size = hidden_size  # Embedding dimension for each patch
        self.intermediate_size = intermediate_size  # Dimension for MLP layer in transformer blocks
        self.num_hidden_layers = num_hidden_layers  # Number of transformer encoder layers
        self.num_attention_heads = num_attention_heads  # Number of attention heads in multi-head attention
        self.num_channels = num_channels  # Number of channels in input images (e.g., 3 for RGB)
        self.patch_size = patch_size  # Size of each patch extracted from the image
        self.image_size = image_size  # Size of the input image (assumed square)
        self.attention_dropout = attention_dropout  # Dropout rate for attention layers
        self.layer_norm_eps = layer_norm_eps  # Epsilon value for layer normalization
        self.num_image_tokens = num_image_tokens  # Number of tokens after patch embedding


class SiglipVisionEmbeddings(nn.Module):
    """
    Embedding layer for the Vision Transformer (ViT). This class divides an image into patches, 
    applies a convolution to transform patches into embeddings, and adds positional embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # The output dimension for each patch
        self.image_size = config.image_size  # Size of the input image
        self.patch_size = config.patch_size  # Size of each patch

        # Convolutional layer that acts as a patch embedding. It extracts patches and projects them to embedding space.
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,  # Number of input channels (e.g., 3 for RGB)
            out_channels=self.embed_dim,  # Output dimension for each patch embedding
            kernel_size=self.patch_size,  # Kernel size equal to patch size
            stride=self.patch_size,  # Stride equal to patch size for non-overlapping patches
            padding="valid",  # No padding added
        )

        # Calculate the number of patches in each image (assuming square patches and image dimensions)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        # Positional embedding to retain spatial information for each patch
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def visualize_patches(self, pixel_values: torch.FloatTensor):
        """
        Visualizes the input images as heatmaps to show the channel intensity. 
        This function helps to understand the input tensor's shape and its visual representation.
        """
        batch_size, channels, height, width = pixel_values.shape
        fig, axes = plt.subplots(1, batch_size, figsize=(12, 6))
        for i in range(batch_size):
            sns.heatmap(pixel_values[i][0].cpu().detach().numpy(), ax=axes[i], cmap="viridis")
            axes[i].set_title(f"Image {i + 1} - Channel 1")
        plt.show()

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass to embed image patches and add positional encodings.
        Args:
            pixel_values (torch.FloatTensor): Input tensor of shape [Batch_Size, Channels, Height, Width].
        Returns:
            torch.Tensor: Patch embeddings with positional encodings added, shape [Batch_Size, Num_Patches, Embed_Dim].
        """
        self.visualize_patches(pixel_values)
        _, _, height, width = pixel_values.shape  # [Batch_Size, Channels, Height, Width]

        # Apply convolution to extract and embed patches
        # Output shape: [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        patch_embeds = self.patch_embedding(pixel_values)

        # Flatten the patches to [Batch_Size, Embed_Dim, Num_Patches]
        embeddings = patch_embeds.flatten(2)
                # Transpose to [Batch_Size, Num_Patches, Embed_Dim] for compatibility with transformer layers
        embeddings = embeddings.transpose(1, 2)

        # Add positional encodings to the patch embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)

        return embeddings


class SiglipAttention(nn.Module):
    """
    Multi-headed attention module. This is based on the 'Attention Is All You Need' paper.
    It calculates the attention scores and updates the hidden states accordingly.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # Total embedding dimension
        self.num_heads = config.num_attention_heads  # Number of attention heads
        self.head_dim = self.embed_dim // self.num_heads  # Dimension per attention head
        self.scale = self.head_dim**-0.5  # Scaling factor for attention scores
        self.dropout = config.attention_dropout  # Dropout rate for attention weights

        # Linear layers to project the input into queries, keys, and values for attention
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-headed self-attention.
        Args:
            hidden_states (torch.Tensor): Input tensor of shape [Batch_Size, Num_Patches, Embed_Dim].
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Attention output and attention weights.
        """
        # Project hidden states to queries, keys, and values
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)  # [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)  # [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)  # [Batch_Size, Num_Patches, Embed_Dim]

        # Reshape for multi-headed attention: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores (Q * K^T / sqrt(d_k))
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Validate attention weights' dimensions
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is {attn_weights.size()}")

        # Apply softmax to get probabilities and dropout (only during training)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Multiply attention weights with value states to get the attention output
        attn_output = torch.matmul(attn_weights, value_states)

        # Validate attention output's dimensions
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is {attn_output.size()}")

        # Reshape back to [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)  # Project back to embedding dimension

        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # First fully connected layer transforming the hidden size to an intermediate size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # Second fully connected layer transforming the intermediate size back to the hidden size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch_Size, Num_Patches, Embed_Dim] 
        # Applies the first linear transformation
        hidden_states = self.fc1(hidden_states)
        # Applies GELU activation function (Gaussian Error Linear Unit) for non-linearity
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # Applies the second linear transformation
        hidden_states = self.fc2(hidden_states)
        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Attention mechanism for the encoder layer
        self.self_attn = SiglipAttention(config)
        # Layer normalization applied before the attention
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        # MLP module for further processing of the hidden states
        self.mlp = SiglipMLP(config)
        # Layer normalization applied before the MLP
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Save the input as the residual for skip connection
        residual = hidden_states
        # Apply layer normalization before the self-attention module
        hidden_states = self.layer_norm1(hidden_states)
        # Pass through the self-attention layer
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # Add the residual connection (skip connection) to maintain information
        hidden_states = residual + hidden_states

        # Save the output as the new residual for the next skip connection
        residual = hidden_states
        # Apply layer normalization before the MLP
        hidden_states = self.layer_norm2(hidden_states)
        # Pass through the MLP
        hidden_states = self.mlp(hidden_states)
        # Add the residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # A list of encoder layers, each consisting of attention and MLP components
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        # Iterate through each encoder layer
        for encoder_layer in self.layers:
            # Pass the hidden states through the encoder layer
            hidden_states = encoder_layer(hidden_states)

        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        # Embeddings module to convert pixel values into patch embeddings
        self.embeddings = SiglipVisionEmbeddings(config)
        # Encoder module consisting of multiple layers
        self.encoder = SiglipEncoder(config)
        # Layer normalization applied after the encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Input shape: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)
        # Pass through the encoder layers
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        # Apply the final layer normalization
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Output shape: [Batch_Size, Num_Patches, Embed_Dim]
        return last_hidden_state


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # The overall vision transformer model
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> torch.Tensor:
        # Input shape: [Batch_Size, Channels, Height, Width]
        # Returns the encoded features from the vision model
        return self.vision_model(pixel_values=pixel_values)
