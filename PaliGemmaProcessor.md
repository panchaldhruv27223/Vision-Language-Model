# PaliGemmaProcessor: Vision-Language Preprocessing Module

This document explains the structure and functionality of the `PaliGemmaProcessor` class and its associated helper functions, as found in the `processing_paligemma.py` file. This processor is used in a Vision-Language Model (VLM) setup inspired by Google's PaliGemma model.

---

## Overview

`PaliGemmaProcessor` prepares inputs for a vision-language model by:

* Preprocessing images (resize, normalize, etc.)
* Inserting visual tokens into text prompts
* Tokenizing the combined prompt
* Returning everything as PyTorch tensors

---

## Class: `PaliGemmaProcessor`

### Purpose

* **Combines image and text preprocessing**
* **Handles special token logic (e.g., , , )**
* **Converts data into model-ready PyTorch tensors**

### Constructor ("`__init__`"):

#### Parameters:

* `tokenizer`: HuggingFace tokenizer
* `num_image_tokens`: Number of visual tokens to insert (e.g., 32)
* `image_size`: Target size to resize images to (e.g., 224)

#### Logic:

1. Adds special tokens to the tokenizer:
   * `<image>` (used to represent image tokens)
   * `<locXXXX>` (used for object detection)
   * `<segXXX>` (used for segmentation)
2. Disables auto-BOS/EOS token addition
3. Stores image token ID

### Forward: `__call__()`

#### Parameters:

* `text`: List of 1 string prompt
* `images`: List of 1 PIL image

#### Steps:

1. Preprocesses image via helper functions
2. Adds `<image>` tokens and `<s>` (BOS) to text prompt
3. Tokenizes updated prompt
4. Converts everything to PyTorch tensors

#### Output:

A dictionary with:

```python
{
  "pixel_values": Tensor [1, 3, H, W],
  "input_ids": Tensor [1, seq_len],
  "attention_mask": Tensor [1, seq_len]
}
```

---

## Helper Functions

### `resize(image, size, resample=None)`

Resizes a PIL image to a fixed (width, height).

### `rescale(image, scale, dtype=np.float32)`

Scales pixel values from [0, 255] → [0, 1] using `scale=1/255.0`.

### `normalize(image, mean, std)`

Normalizes image channels using ImageNet stats:

```python
image = (image - mean) / std
```

### `process_images(images, ...)`

Applies all of the above:

* Resize
* Rescale
* Normalize
* Transpose [H, W, C] → [C, H, W]

### `add_image_tokens_prompt(prompt, bos_token, image_seq_len, image_token)`

Returns a string like:

```
<image><image>...<image><s> Prompt text
```

Used to simulate image embeddings within token stream.

---

## Special Tokens

* `<image>`: Repeated before prompt to reserve space for visual features.
* `<locXXXX>`: 1024 object detection tags
* `<segXXX>`: 128 segmentation labels

---

## Final Pipeline

```
Image (PIL) ──▶ Resize ──▶ Rescale ──▶ Normalize ──▶ Transpose ─▶ Torch Tensor
     │
     ▼
Text Prompt ───────▶ Add <image> tokens ──▶ Tokenize ─▶ Torch Tensor

Final Output:
{
  "pixel_values": [1, 3, 224, 224],
  "input_ids": [1, seq_len],
  "attention_mask": [1, seq_len]
}
```
