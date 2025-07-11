### 1. **KVCache**

* **Purpose:** Stores key and value tensors for attention during inference (used for caching past keys/values in autoregressive models).
* **Methods:**
  * `<span>num_items()</span>`: Returns the length of the cache (sequence length).
  * `<span>update()</span>`: Concatenates new keys/values to existing cache per layer.


---

### 2. **repeat\_kv**

* **Purpose:** Repeats key/value heads to match the number of attention heads for grouped attention.


---

### 3. **GemmaConfig**

* **Purpose:** Holds hyperparameters for the Gemma language model (hidden size, vocab size, etc.).

---

### 4. **PaliGemmaConfig**

* **Purpose:** Configuration wrapper for the full vision-language model.
* **Combines:** Vision config (`<span>SiglipVisionConfig</span>`) and text config (`<span>GemmaConfig</span>`).
* **Adds:** Projection dimensions, special image tokens, etc.

---

### 5. **GemmaRMSNorm**

* **Purpose:** Implements RMS (Root Mean Square) normalization.
* **Details:** Normalizes input without subtracting mean.

---

### 6. **GemmaMLP**

* **Purpose:** Feedforward MLP used after attention in transformer blocks.
* **Details:** Uses GELU non-linearity and three linear layers (gate, up, down).

---

### 7. **GemmaRotaryEmbeddings**

* **Purpose:** Implements Rotary Positional Embeddings (RoPE) for attention.
* **Returns:** Cosine and sine values used to apply position information.

---

### 8. **rotate\_half & apply\_rotary\_pos\_emb**

* **Purpose:** Helper functions for rotating and applying RoPE to query and key.

---

### 9. **GemmaAttention**

* **Purpose:** Implements multi-head self-attention with rotary positional encoding.
* **Supports:** Caching (kv\_cache), grouped key/value heads, and causal masking.
* **Process:**
  * Projects Q, K, V
  * Applies RoPE
  * Applies mask and computes attention weights
  * Returns attention output

---

### 10. **GemmaDecoderLayer**

* **Purpose:** One transformer block of the Gemma decoder.
* **Components:**
  * Self-attention (with RoPE)
  * MLP
  * RMS Norm
  * Residual connections

---

### 11. **GemmaModel**

* **Purpose:** Stacks multiple `<span>GemmaDecoderLayer</span>` to build full language model.
* **Returns:** Hidden states from top layer.

---

### 12. **GemmaForCausalLM**

* **Purpose:** Wraps `<span>GemmaModel</span>` with a language modeling head for prediction.
* **Adds:**
  * Output projection (`<span>lm_head</span>`) to vocab size
  * Optional weight tying

---

### 13. **PaliGemmaMultiModelProjector**

* **Purpose:** Projects visual features from vision encoder to the same dimension as text model input.
* **Input:** Vision features (from `<span>SiglipVisionModel</span>`)
* **Output:** Aligned embeddings for fusion with text.

---

### 14. **PaliGemmaForConditionalGeneration**

* **Purpose:** Complete vision-language model combining SigLIP vision encoder + Gemma language model.
* **Components:**
  * `<span>vision_tower</span>`: Vision model (`<span>SiglipVisionModel</span>`)
  * `<span>multi_model_projector</span>`: Projects visual embeddings to language model space
  * `<span>language_model</span>`: `<span>GemmaForCausalLM</span>`

#### Method: `<span>_merge_input_ids_with_image_features()</span>`

* Injects image features into input embeddings where `<span><image></span>` token appears
* Handles positional encoding and attention masks

#### Method: `<span>forward()</span>`

* Pipeline:
  * Get input embeddings from token IDs
  * Pass image to vision model → get features
  * Project image features to text hidden dim
  * Merge image features and input text tokens
  * Pass to `<span>GemmaForCausalLM</span>`
  * Output logits (and optionally KV cache)
