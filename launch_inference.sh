MODEL_PATH = "C:\Users\dhruv\Documents\NLP\Vision_language_model\paligemma2-3b-pt-224"
PROMPT = "this car "
IMAGE_FILE_PATH = 
MAX_TOKENS_TO_GENERATE = 100
TEMPERATURE = 0.8
TOP_P = 0.9
DO_SAMPLE = "FALSE"
ONLY_CPU = "FALSE"


python inference.py \
    --model_path "$MODEL_PATH" \
    --prompt "$PROMPT" \
    --image_file_path "$IMAGE_FILE_PATH" \
    --max_tokens_to_generate $MAX_TOKENS_TO_GENERATE \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --do_sample $DO_SAMPLE \
    --only_cpu $ONLY_CPU \

    