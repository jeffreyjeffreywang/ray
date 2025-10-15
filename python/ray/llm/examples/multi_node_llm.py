import ray

from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

# config = vLLMEngineProcessorConfig(
#     model_source="Qwen/Qwen3-0.6B",
#     apply_chat_template=True,
#     concurrency=1,
#     batch_size=64,
# )

# processor = build_llm_processor(
#     config,
#     preprocess=lambda row: dict(
#         messages=[
#             {"role": "user", "content": row["prompt"]},
#         ],
#         sampling_params=dict(
#             temperature=0.6,
#             max_tokens=100,
#         ),
#     ),
#     builder_kwargs=dict(
#         chat_template_kwargs={"enable_thinking": True},
#     ),
# )

# ds = ray.data.from_items([{"prompt": "What is 2+2?"}])
# ds = processor(ds)
# for row in ds.take_all():
#     print(row)

config = vLLMEngineProcessorConfig(
    model_source="facebook/opt-1.3b",
    engine_kwargs=dict(
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        # pipeline_parallel_size=2,
        # tensor_parallel_size=2,
        distributed_executor_backend="ray",
    ),
    concurrency=1,
    batch_size=64,
    apply_chat_template=False,
    tokenize=False,
    detokenize=False,
    accelerator_type="A10G",
    placement_group_config=dict(
        bundles=[
            {"CPU": 1, "accelerator_type:A10G": 1, "GPU": 1},
            # {"CPU": 1, "accelerator_type:A10G": 1, "GPU": 1},
        ],
        strategy="PACK",
    ),
)

processor = build_llm_processor(
    config,
    preprocess=lambda row: dict(
        prompt=f"You are a calculator. {row['id']} ** 3 = ?",
        sampling_params=dict(
            temperature=0.3,
            max_tokens=20,
            detokenize=True,
        ),
    ),
    postprocess=lambda row: dict(
        resp=row["generated_text"],
        **row,  # This will return all the original columns in the dataset.
    ),
)

ds = ray.data.range(5)
ds = processor(ds)
for row in ds.take_all():
    print(row)