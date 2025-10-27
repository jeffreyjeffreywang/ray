import io

import PIL.Image
import requests

import ray
from ray.data.llm import (
    MultimodalProcessorConfig,
    build_llm_processor,
    vLLMEngineProcessorConfig,
)

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"

# Download image from URL and convert to PIL Image
response = requests.get(image_url)
image_pil = PIL.Image.open(io.BytesIO(response.content))

# Create dataset first
ds = ray.data.range(1)
ds = ds.map(lambda x: {"id": x["id"], "val": x["id"] + 10})

multimodal_processor_config = MultimodalProcessorConfig(
    model=model_path,
    concurrency=1,
)

multimodal_processor = build_llm_processor(
    multimodal_processor_config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this video in {row['val']} words.",
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": image_url},
                    # }
                    {
                        "type": "image_pil",
                        "image_pil": image_pil,
                    },
                ],
            },
        ],
    ),
)

processor_config = vLLMEngineProcessorConfig(
    model_source=model_path,
    task_type="generate",
    engine_kwargs=dict(
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    ),
    apply_chat_template=True,
    tokenize=False,
    detokenize=False,
    batch_size=16,
    concurrency=1,
)

processor = build_llm_processor(
    processor_config,
    preprocess=lambda row: dict(
        sampling_params=dict(
            temperature=0.3,
            max_tokens=50,
        ),
    ),
)

ds = multimodal_processor(ds)
ds = processor(ds)
ds = ds.materialize()
outs = ds.take_all()

# print(outs[0]["mm_data"])
print(outs[0]["generated_text"])
