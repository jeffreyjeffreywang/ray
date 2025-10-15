from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

"""
model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
video_path = "https://content.pexels.com/videos/free-videos.mp4"

llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    limit_mm_per_prompt={"video": 1},
)

sampling_params = SamplingParams(
    max_tokens=1024,
)

video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
            {"type": "text", "text": "describe this video."},
            {
                "type": "video",
                "video": video_path,
                "total_pixels": 20480 * 28 * 28,
                "min_pixels": 16 * 28 * 28
            }
        ]
    },
]

messages = video_messages
processor = AutoProcessor.from_pretrained(model_path)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(messages)
mm_data = {}
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)
"""

import ray

from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
video_path = "https://content.pexels.com/videos/free-videos.mp4"

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
    has_video=True,
)

processor = build_llm_processor(
    processor_config,
    preprocess=lambda row: dict(
        model=model_path,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                    {"type": "text", "text": f"Describe this video {row['id']} words."},
                    {
                        "type": "video",
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28,
                        "min_pixels": 16 * 28 * 28
                    }
                ]
            },
        ],
        sampling_params=dict(
            temperature=0.3,
            max_tokens=50,
        ),
    ),
)

ds = ray.data.range(1)
ds = ds.map(lambda x: {"id": x["id"], "val": x["id"] + 5})
ds = processor(ds)
ds = ds.materialize()
outs = ds.take_all()
print(outs[0]["generated_text"])