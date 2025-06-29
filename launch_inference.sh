#!/bin/bash

MODEL_PATH="/home/dhruv/Documents/Dhruv_Vision_language_model/paligemma-3b-pt-224"
PROMPT="which animal is in the image?"
IMAGE_FILE_PATH="/home/dhruv/Documents/Dhruv_Vision_language_model/Vision-Language-Model/test_images/samuel-scrimshaw-KeUKM5N-e_g-unsplash.jpg"
MAX_TOKENS_TO_GENERATE=100
TEMPERATURE=0.8
TOP_P=0.9
DO_SAMPLE="FALSE"
ONLY_CPU="FALSE"

# echo "Using model: $MODEL_PATH"
# echo "Using image: $IMAGE_FILE_PATH"

python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU 