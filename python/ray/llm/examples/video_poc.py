import asyncio

from transformers import AutoProcessor

import vllm
from vllm import AsyncLLMEngine, SamplingParams
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import MultiModalContentParser, MultiModalItemTracker

model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
video_path = "https://content.pexels.com/videos/free-videos.mp4"

async def main():
    engine_args = vllm.AsyncEngineArgs(
        model=model_path,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1},
    )

    llm = AsyncLLMEngine.from_engine_args(engine_args)

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

    # image_inputs, video_inputs = process_vision_info(messages)
    # mm_data = {}
    # if video_inputs is not None:
    #     mm_data["video"] = video_inputs

    model_config = ModelConfig(model=model_path)

    tracker = MultiModalItemTracker(model_config, None)
    parser = MultiModalContentParser(tracker)
    parser.parse_video(video_path)

    mm_data = tracker.all_mm_data()

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": {
            "min_pixels": 16 * 28 * 28,
        }
    }
    llm_prompt = vllm.inputs.data.TextPrompt(
        prompt=llm_inputs["prompt"],
        multi_modal_data=llm_inputs["multi_modal_data"],
    )

    # Generate text asynchronously
    request_id = "video_request_1"
    results_generator = llm.generate(llm_prompt, sampling_params, request_id)
    
    async for request_output in results_generator:
        if request_output.finished:
            generated_text = request_output.outputs[0].text
            print("Generated text:", generated_text)
            break

if __name__ == "__main__":
    asyncio.run(main())
