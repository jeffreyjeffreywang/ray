import ray

# ray.init(runtime_env={"GLOO_SOCKET_IFNAME": "ens5"})

# # Check resources
# resources = ray.cluster_resources()
# print(f"Available GPUs: {resources.get('GPU', 0)}")
# print(f"Available CPUs: {resources.get('CPU', 0)}")

# # Check nodes and print IP addresses
# nodes = ray.nodes()
# print(f"\n--- Cluster Nodes ---")
# print(f"Number of nodes: {len(nodes)}")
# print(f"Nodes: {nodes}")

# # Get cluster info for more details
# cluster_resources = ray.cluster_resources()
# print(f"Cluster resources: {cluster_resources}")

from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

config = vLLMEngineProcessorConfig(
    model_source="facebook/opt-1.3b",
    engine_kwargs=dict(
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
        # pipeline_parallel_size=2,
        tensor_parallel_size=2,
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
            {"CPU": 1, "accelerator_type:A10G": 1, "GPU": 1},
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
