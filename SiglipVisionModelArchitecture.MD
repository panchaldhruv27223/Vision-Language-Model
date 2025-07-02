$$
Class Breakdown
$$

### 1️⃣ **SiglipVisionConfig**

- Stores the hyperparameters and architecture configuration.
- Includes:
  
  - `hidden_size`: embedding dimension
  - `intermediate_size`: MLP hidden layer size
  - `num_attention_heads`: number of attention heads
  - `num_hidden_layers`: number of transformer layers
  - `image_size`: input image size
  - `patch_size`: patch size for image splitting
  - Other hyperparameters like dropout and layer norm epsilon

---

### 2️⃣ **SiglipVisionEmbeddings**

- Converts an image into a sequence of patch embeddings.
- Adds positional embeddings.
- Internally uses a Conv2D layer with:
  - `kernel_size = patch_size`
  - `stride = patch_size`
- Output shape: `[batch_size, num_patches, hidden_size]`

---

### 3️⃣ **SiglipMLP**

- Standard MLP block used inside the transformer layer.
- Structure:
  - Linear → GELU → Linear
- Expands from `hidden_size` → `intermediate_size` → `hidden_size`.

---

### 4️⃣ **SiglipAttention**

- Implements multi-head self-attention.
- Components:
  - Query, Key, Value projections.
  - Scaled dot-product attention.
  - Output projection linear layer.
- Handles multiple attention heads with scaling and dropout.

---

### 5️⃣ **SiglipEncoderLayer**

- A single transformer encoder block.
- Consists of:
  - LayerNorm → Self-Attention → Residual Add
  - LayerNorm → MLP → Residual Add
- Pre-Norm architecture improves training stability.

---

### 6️⃣ **SiglipEncoder**

- A stack of multiple encoder layers (`num_hidden_layers`).
- Processes the input embedding sequence.
- Outputs transformed embeddings.

---

### 7️⃣ **SiglipVisionTransformer**

- Combines:
  - Embedding layer (`SiglipVisionEmbeddings`)
  - Transformer encoder (`SiglipEncoder`)
  - Final LayerNorm
- Produces the final token embeddings for the input image.

---

### 8️⃣ **SiglipVisionModel**

- High-level wrapper over `SiglipVisionTransformer`.
- Entry point for users to process image tensors.
- Returns the sequence of embeddings for image patches.

---

## Output Shapes

| Stage            | Shape                         |
| ---------------- | ----------------------------- |
| Input Image      | `[B, 3, H, W]`                |
| Patch Embeddings | `[B, N_patches, hidden_size]` |
| Encoder Output   | `[B, N_patches, hidden_size]` |
| Final Output     | `[B, N_patches, hidden_size]` |

---

## Example Usage

```python
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import torch

# Create config
config = SiglipVisionConfig(
    hidden_size=768,
    num_hidden_layers=4,
    num_attention_heads=12,
    intermediate_size=3072,
    image_size=224,
    patch_size=16,
)

# Initialize model
model = SiglipVisionModel(config)

# Dummy image input
dummy_image = torch.randn(2, 3, 224, 224)  # Batch of 2 images

# Forward pass
output = model(dummy_image)

print("Output shape:", output.shape)
```

## Author

## $$ DHRUV PANCHAL  $$


