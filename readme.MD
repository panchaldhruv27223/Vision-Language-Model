# Vision-Language Model (PaliGemma)

This repository contains a **from-scratch PyTorch implementation** of a custom Vision-Language Model inspired by Google's **PaliGemma** architecture. It integrates a **SigLIP-based vision encoder** with a **Gemma-based causal language decoder**, creating a multimodal architecture capable of processing both images and text.

---

## 🚀 Project Highlights

* **Vision Encoder**: Custom implementation of [SigLIP](https://arxiv.org/abs/2306.10883)-style transformer.
* **Language Decoder**: Custom Gemma-style causal language model with rotary embeddings and KV caching.
* **Multimodal Fusion**: Merges image embeddings as tokens in the input sequence.
* **Inference Script**: Simple interface for running inference on image-text pairs.

---

## 🏗️ Architecture Overview

![Architecture Diagram](vlm_paligemma_model.png)


---

---

## 📦 Dependencies

* Python 3.10+
* PyTorch
* torchvision
* transformers
* fire
* PIL

Install dependencies:

```bash
pip install torch torchvision transformers fire pillow
```

---

## 🧠 Inspiration & References

This project is heavily inspired by Google DeepMind's **PaliGemma** Vision-Language architecture:

* 🔗 [Google Blog – PaliGemma Explained](https://developers.googleblog.com/en/gemma-explained-paligemma-architecture/)
* 📺 [YouTube Tutorial](https://youtu.be/vAmKB7iPkWw?si=hJ51teOwZ6Y3Zjct)
* 📄 [Official Paper (arXiv:2407.07726)](https://arxiv.org/abs/2407.07726)

> This is a **research and learning project**. The implementation is completely custom and does not use pre-trained models.


---

## 📜 License

This project is currently unlicensed. Feel free to fork and experiment for educational purposes. Contact the author if you plan to use this commercially.

---

## 👤 Author

**Dhruv Panchal**
[GitHub](https://github.com/panchaldhruv27223)

Feel free to star ⭐ this repo if you find it useful!
