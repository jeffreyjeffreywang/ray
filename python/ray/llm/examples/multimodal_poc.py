import ray
from ray.data.llm import (
    MultimodalProcessorConfig,
    build_llm_processor,
    vLLMEngineProcessorConfig,
)

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
video_path = "https://content.pexels.com/videos/free-videos.mp4"

# Create dataset first
ds = ray.data.range(1)
ds = ds.map(lambda x: {"id": x["id"], "val": x["id"] + 10})

multimodal_processor_config = MultimodalProcessorConfig(
    model_path=model_path,
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
                    {
                        "type": "video_url",
                        "video_url": {"url": video_path},
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
        limit_mm_per_prompt={"video": 1},
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
