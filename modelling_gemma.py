import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class KVCache():
    """
    Key-Value Cache for transformer models to store previous attention computations.
    This optimization prevents recomputing attention for tokens we've already processed,
    making text generation more efficient.
    """
    
    def __init__(self) -> None:
        """
        Initialize empty cache for storing key and value states.
        
        key_cache: Stores key tensors for each transformer layer
        value_cache: Stores corresponding value tensors for each layer
        Both are implemented as lists where each index corresponds to a layer
        """
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
   
    def num_items(self) -> int:
        """
        Returns the number of tokens currently stored in the cache.
        
        Returns:
            int: Number of cached tokens (sequence length) or 0 if cache is empty
            
        Note: We only need to check key_cache since key and value caches always have same length
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            # We return Seq_Len which is the third dimension (index -2)
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,    # New key states to be added
        value_states: torch.Tensor,  # New value states to be added
        layer_idx: int,              # Index of the transformer layer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new key and value states for a specific layer.
        
        Args:
            key_states: New key tensors to add to cache
            value_states: New value tensors to add to cache
            layer_idx: Which transformer layer these states belong to
            
        Returns:
            Tuple containing:
            - Complete key states (cached + new) for the layer
            - Complete value states (cached + new) for the layer
            
        Shape of tensors:
        - key_states: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        - value_states: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        """
        
        if len(self.key_cache) <= layer_idx:
            # First time we're seeing this layer - initialize its cache
            # Simply store the new states as they are
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # We already have cached states for this layer
            # Concatenate new states with cached states along sequence length dimension
            # This effectively extends our memory of previous tokens
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx], key_states], 
                dim=-2  # Concatenate along sequence length dimension
            )
            self.value_cache[layer_idx] = torch.cat(
                [self.value_cache[layer_idx], value_states], 
                dim=-2  # Concatenate along sequence length dimension
            )
            
        # Return the complete states (cached + new) for this layer
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE) for transformer models.
    RoPE encodes position information directly into the attention computation
    through rotation matrices, allowing better handling of relative positions.
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        """
        Initialize the RoPE module.
        
        Args:
            dim: Hidden dimension size (head_dim in attention)
            max_position_embeddings: Maximum sequence length to support
            base: Base for the frequency calculations (affects how position information scales)
        """
        super().__init__()
        self.dim = dim  # Set to the head_dim from attention
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Calculate frequencies for rotation
        # Formula: θᵢ = base^(-2i/dim) where i = 0, 1, ..., dim/2 - 1
        # These frequencies determine how fast each dimension rotates with position
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        
        # Register frequencies as a buffer (not a parameter)
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        Compute the cos and sin values for rotary embeddings.
        
        Args:
            x: Input tensor [batch_size, num_attention_heads, seq_len, head_size]
            position_ids: Position indices [batch_size, seq_len]
            seq_len: Optional sequence length
            
        Returns:
            cos, sin tensors for rotating query and key vectors
        """
        # Ensure frequencies are on same device as input
        self.inv_freq.to(x.device)
        
        # Expand frequencies for batch processing
        # Shape: [batch_size, dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        
        # Expand position IDs
        # Shape: [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        # Handle different device types (especially for Apple Silicon)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        # Compute rotation frequencies for each position
        with torch.autocast(device_type=device_type, enabled=False):
            # Matrix multiply to get frequencies for each position
            # Shape: [batch_size, seq_len, dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            
            # Duplicate frequencies for full dimension
            # Shape: [batch_size, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            
            # Compute cos and sin values
            # Shape: [batch_size, seq_len, dim]
            cos = emb.cos()
            sin = emb.sin()
            
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """
    Helper function to create interleaved negative-positive pairs.
    For a vector [x1, x2, x3, x4], returns [-x2, x1, -x4, x3].
    
    This is used in RoPE to create the rotation effect.
    """
    # Split the last dimension in half
    x1 = x[..., : x.shape[-1] // 2]  # First half
    x2 = x[..., x.shape[-1] // 2 :]  # Second half
    # Concatenate with alternating signs
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary position embeddings to query and key tensors.
    
    The rotation is applied using the formula:
    [cos θ  -sin θ] [x]
    [sin θ   cos θ] [y]
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine of rotation angles
        sin: Sine of rotation angles
        unsqueeze_dim: Dimension to add for broadcasting
        
    Returns:
        Rotated query and key tensors
    """
    # Add head dimension for broadcasting
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    # Apply rotation using the formula from RoPE paper
    # For each vector [x, y]:
    # x_rotated = x * cos - y * sin
    # y_rotated = x * sin + y * cos
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

# Theory: Grouped Query Attention (GQA)
# GQA is an optimization of the standard attention mechanism where we use fewer key-value heads
# than query heads. This reduces memory usage and computational cost while maintaining model quality.
# Each key-value head is shared across multiple query heads, implemented through the repeat_kv function.

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeats key-value heads to match the number of query heads in Grouped Query Attention.
    
    Args:
        hidden_states: Tensor of shape [batch, num_kv_heads, seq_len, head_dim]
        n_rep: Number of times to repeat each key-value head
    
    Returns:
        Tensor of shape [batch, num_kv_heads * n_rep, seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:  # If no repetition needed, return as is
        return hidden_states
    # Add a new dimension and expand (repeat) along it
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    # Reshape to combine the kv_heads and repetition dimensions
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# Theory: Multi-head Attention
# Multi-head attention allows the model to jointly attend to information from different 
# representation subspaces at different positions. Each head can focus on different aspects
# of the input, making the model more expressive.

class GemmaAttention(nn.Module):
    """
    Implements the attention mechanism for the Gemma model, using Grouped Query Attention (GQA)
    and Rotary Position Embeddings (RoPE).
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        """
        Initialize the attention module with the given configuration.
        
        Args:
            config: Configuration object containing model hyperparameters
            layer_idx: Index of this layer in the transformer stack
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Extract configuration parameters
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size  # Total size of the hidden representations
        self.num_heads = config.num_attention_heads  # Number of query heads
        self.head_dim = config.head_dim  # Dimension of each attention head
        self.num_key_value_heads = config.num_key_value_heads  # Number of key-value heads (fewer than query heads)
        # Calculate how many query heads share each key-value head
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta  # Base for rotary position embeddings
        self.is_causal = True  # Use causal attention mask (can't look at future tokens)

        # Verify that the hidden size is divisible by the number of heads
        assert self.hidden_size % self.num_heads == 0            

        # Initialize the projection matrices
        # Q, K, V projections transform the input hidden states into query, key, and value representations
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # Output projection combines the attention outputs back to hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        # Initialize rotary position embeddings
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of the attention mechanism.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Mask to prevent attention to certain positions
            position_ids: Indices of positions for rotary embeddings
            kv_cache: Optional cache for key and value states in inference
        """
        # Get batch size, sequence length, and hidden size
        bsz, q_len, _ = hidden_states.size()

        # 1. Project input hidden states to query, key, and value states
        query_states = self.q_proj(hidden_states)  # [batch_size, seq_len, num_heads * head_dim]
        key_states = self.k_proj(hidden_states)    # [batch_size, seq_len, num_kv_heads * head_dim]
        value_states = self.v_proj(hidden_states)  # [batch_size, seq_len, num_kv_heads * head_dim]

        # 2. Reshape and transpose for attention computation
        # Separate heads and move head dimension before sequence length
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. Apply rotary position embeddings (RoPE)
        # RoPE provides relative positional information by rotating vectors in complex space
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. Use cached key-value states if provided (for efficient inference)
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # 5. Implement Grouped Query Attention
        # Repeat key-value heads to match number of query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 6. Compute attention scores
        # Scaled dot-product attention: Q * K^T / sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 7. Apply attention mask
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # 8. Apply softmax to get attention probabilities
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 9. Apply attention dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 10. Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)

        # 11. Verify output shape
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        # 12. Reshape attention output
        # Transpose and reshape to combine all heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # 13. Project back to model dimension
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    # Ignore copy
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states

class GemmaForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()

    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, kv_cache: Optional[KVCache] = None
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        # Shape: [Batch_Size, Seq_Len, Hidden_Size]
        scaled_image_features = image_features / (self.config.hidden_size**0.5)
    
        # Combine the embeddings of the image tokens, the text tokens and mask out all the padding tokens.
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        # Shape: [Batch_Size, Seq_Len]. True for text tokens
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        # Shape: [Batch_Size, Seq_Len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index
        # Shape: [Batch_Size, Seq_Len]. True for padding tokens
        pad_mask = input_ids == self.pad_token_id

        # We need to expand the masks to the embedding dimension otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Add the text embeddings
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)
        # Insert image embeddings. We can't use torch.where because the sequence length of scaled_image_features is not equal to the sequence length of the final embedding
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        # Zero out padding tokens
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        #### CREATE THE ATTENTION MASK ####

        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens. 
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # Make sure the input is right-padded
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. Extra the input embeddings
        # shape: (Batch_Size, Seq_Len, Hidden_Size)
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. Merge text and images
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Hidden_Size]
        image_features = self.multi_modal_projector(selected_image_feature)

        # Merge the embeddings of the text tokens and the image tokens
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)
        
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs