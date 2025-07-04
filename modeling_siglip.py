import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module


class SiglipVisionConfig:

    def __init__(
        self,
        ### hidden size in the NN (embedding size )
        hidden_size = 768,
        ### number of layers in the NN
        intermediate_size = 3072,
        ## number of heads
        num_attention_heads = 12,
        ## number of layers in the transformer
        num_hidden_layers = 12,
        num_channels = 3,
        image_size = 224,
        patch_size = 16,
        layer_norm_eps = 1e-6,
        attention_dropout = 0.0,
        num_image_tokens = None,
        **kwargs
    ):
        
        super().__init__()


        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbeddings(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels = config.num_channels,
            out_channels = self.embed_dim,
            kernel_size = config.patch_size,
            stride = self.patch_size,
            padding = "valid"
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.positional_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1,-1)),
            persistent = False,
        )
        
    
    def forward(self, pixel_values):
        # batch_size and number_Channel
        _, _, height, width = pixel_values.shape

        patch_embeds = self.patch_embedding(pixel_values)
        # (batch_size, embed_dim, num_patches)
        embeddings = patch_embeds.flatten(2)
        
        embeddings = embeddings.transpose(1,2)

        embeddings = embeddings + self.position_embeddings(self.position_ids)

        return embeddings


class SiglipMLP(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate = 'tanh')
        hidden_states = self.fc2(hidden_states)

        return hidden_states

class SiglipAttention(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim , self.embed_dim)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)

        key_states = self.k_proj(hidden_states)

        values_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        values_states = values_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        attn_weight = (torch.matmul(query_states, key_states.transpose(2,3)) * self.scale)

        if attn_weight.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weight should be of the shape {(batch_size, self.num_heads, seq_len, seq_len)} but it is"
            f"{attn_weight.size()}" 
            )
        

        attn_weight = nn.functional.softmax(attn_weight, dim= -1, dtype = torch.float32).to(query_states.dtype)

        attn_weight = nn.functional.dropout(attn_weight, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weight, values_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should be shape of {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f"{attn_output.size()}"
            )

        attn_output = attn_output.transpose(1,2).contiguous()

        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weight
        


class SiglipEncoderLayer(Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps = config.layer_norm_eps)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states)
        
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states



class SiglipEncoder(Module):
    def __init__(config):
        super().__init__()
        
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self,inputs_embeds):
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states)

        return hidden_states
        


class SiglipVisionTransformer(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps = config.layer_norm_eps)
    
    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state



class SiglipVisionModel(Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
         
    def forward(self, pixel_values):
        ## take in the form [batch_size, channel, height, width] --> [batch_size, number_patch, embedding_dim]
        return self.vision_model(pixel_values)


if __name__ == "__main__":
    print("Dhruv It is working fine")