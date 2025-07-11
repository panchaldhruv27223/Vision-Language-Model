import numpy as np
from PIL import Image
import os
import torch 
from typing import List, Dict, Optional, Tuple, Union, Iterable


IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


def add_image_tokens_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def resize(image, size, resample = None, reducing_gap= None ):
    height, width = size[0], size[1]

    resized_image = image.resize((width, height), resample=resample)
    
    return resized_image


def rescale(image, scale, dtype=np.float32):

    rescaled_image = image*scale

    rescaled_image = rescaled_image.astype(dtype)

    return rescaled_image

def normalize(image, mean, std):
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)

    image = (image - mean)/std

    return image



def process_images(images, size, resample, rescale_factor, image_mean, image_std):
    height, width = size[0], size[1]

    images = [resize(image=image, size=(height, width),resample=resample) for image in images]

    images = [np.array(image) for image in images]

    images = [rescale(image, scale= rescale_factor) for image in images]

    images = [normalize(image, mean = image_mean, std = image_std) for image in images]

    images = [image.transpose(2,0,1) for image in images]

    return images


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens, image_size):

        super().__init__()

        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        ## additional token added to the tokenizer
        token_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}

        tokenizer.add_special_tokens(token_to_add)

        ## Extra token for the image

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ] ## these tokens are used for object detection (bounding boxes)

        EXTRA_TOKENS += [
            f"<seg{i:03d}>" for i in range(128)
        ] ## these tokens are used for segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        ## we will add the bos and eos token ourselves

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(self, text, images, padding ="longest", truncation = True):
        assert len(images) == 1 and len(text) == 1,f"Received {len(images)} images for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1 / 255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD 
        )

        pixel_values = np.stack(pixel_values, axis=0)
        pixel_values = torch.tensor(pixel_values)

        ## Prepend a self.image_seq_length number of image tokens to the prompt

        input_string = [
            add_image_tokens_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_length,
                image_token = self.IMAGE_TOKEN
            )
            for prompt in text
        ]


        ## return the input_ids and attention_mask as pytorch tensors

        inputs = self.tokenizer(
            input_string,
            return_tensors = "pt",
            padding = padding,
            truncation = truncation
        )

        return_data = {"pixel_values":pixel_values, **inputs}

        return return_data