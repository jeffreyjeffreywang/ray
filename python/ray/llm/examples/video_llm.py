import ray
from ray.data.llm import MultimodalProcessorConfig, build_llm_processor
from ray.llm._internal.batch.stages.configs import PrepareMultimodalStageConfig

config = MultimodalProcessorConfig(
    model_source="Qwen/Qwen2.5-VL-3B-Instruct",
    prepare_multimodal_stage=PrepareMultimodalStageConfig(
        enabled=True,
    ),
    concurrency=1,
)
processor = build_llm_processor(
    config,
    preprocess=lambda row: dict(
        messages=[
            {"role": "system", "content": "You are a helpful video summarizer."},
            {"role": "user", "content": [
                    {"type": "text", "text": f"Describe this video in {row['id']} sentences."},
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://content.pexels.com/videos/free-videos.mp4"},
                    }
                ]
            },
        ],
    ),
)

ds = ray.data.range(10)
ds = processor(ds)
for row in ds.take_all():
    print(row)