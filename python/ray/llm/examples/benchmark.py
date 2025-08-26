#!/usr/bin/env python3
"""
DISCLAIMER: This script is NOT used for production. It may not align with code styles of the rest of the codebase.

Benchmark script for comparing ServeDeploymentStage vs vLLMEngineStage performance.
Uses 10000 samples from the ShareGPT dataset from Hugging Face.

Run with: python benchmark.py --mode serve_deployment --batch-size 64 --concurrency 2
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import ray
from datasets import load_dataset, load_from_disk
from ray import serve
from ray.data.llm import ServeDeploymentProcessorConfig, build_llm_processor
from ray.llm._internal.batch.processor.vllm_engine_proc import (
    vLLMEngineProcessorConfig,
)
from ray.llm._internal.serve.configs.server_models import LLMEngine
from ray.serve.llm import (
    ChatCompletionRequest,
    CompletionRequest,
    LLMConfig,
    ModelLoadingConfig,
    build_llm_deployment,
)
from ray.serve.schema import ApplicationStatusOverview, ServeStatus

MAX_SAMPLES = 10000
TRUNCATE_PROMPT = 2000
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "max_tokens": 50,
    "top_p": 0.9,
    "ignore_eos": True,  # Ensure a fair comparison by omitting sampling-induced variance
}


@dataclass
class BenchmarkResult:
    stage_type: str
    batch_size: int
    concurrency: int
    total_time: float
    throughput: float
    total_samples: int


def load_sharegpt_dataset() -> List[Dict[str, Any]]:
    """Load entire ShareGPT dataset from local directory or download if not present."""
    dataset_path = "/home/ubuntu/datasets/Code-feedback-sharegpt-renamed"

    print(f"Attempting to load dataset from: {dataset_path}")
    print(f"Path exists: {os.path.exists(dataset_path)}")

    try:
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
        else:
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            dataset = load_dataset(
                "Crystalcareai/Code-feedback-sharegpt-renamed", split="train"
            )
            dataset.save_to_disk(dataset_path)

        data = []

        if len(dataset) > 0:
            print(f"First item structure: {dataset[0]}")
            print(
                f"First item keys: {list(dataset[0].keys()) if hasattr(dataset[0], 'keys') else 'No keys'}"
            )

        for i, item in enumerate(dataset):
            if i >= MAX_SAMPLES:
                break

            # Extract the first human message from the conversation
            messages = item.get("messages", [])
            if messages and len(messages) > 0:
                human_msg = None
                for msg in messages:
                    if msg.get("role") == "human":
                        human_msg = msg.get("value", "")
                        break

                if human_msg:
                    data.append(
                        {
                            "prompt": human_msg,
                            "sampling_params": SAMPLING_PARAMS,
                        }
                    )

        print(
            f"Loaded ShareGPT dataset: {len(data)} samples (limited to {MAX_SAMPLES})"
        )
        assert len(data) > 0, "No data extracted from dataset!"
        return data

    except Exception as e:
        raise RuntimeError(f"Error loading ShareGPT dataset: {e}")


def setup_serve_deployment(args) -> tuple:
    """Set up Ray Serve deployment for benchmarking."""

    deployment_name = "benchmark_deployment"
    app_name = "benchmark_app"

    llm_config = LLMConfig(
        model_loading_config=ModelLoadingConfig(
            model_id="facebook/opt-1.3b",
        ),
        llm_engine=LLMEngine.vLLM,
        accelerator_type="A10G",
        deployment_config=dict(
            name=deployment_name,
            autoscaling_config=dict(
                min_replicas=args.concurrency,
                max_replicas=args.concurrency,
            ),
        ),
        engine_kwargs=dict(
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=4096,
        ),
    )

    llm_app = build_llm_deployment(
        llm_config, override_serve_options={"name": deployment_name}
    )
    serve.run(llm_app, name=app_name)

    print("Waiting for Serve deployment to be ready...")
    max_wait_time = 120
    wait_time = 0
    while not is_app_ready(app_name) and wait_time < max_wait_time:
        time.sleep(5)
        wait_time += 5

    if wait_time >= max_wait_time:
        raise TimeoutError("Deployment failed to become ready within timeout")

    print("Deployment is ready!")
    return deployment_name, app_name


def is_app_ready(app_name: str) -> bool:
    try:
        serve_status: ServeStatus = serve.status()

        if app_name in serve_status.applications:
            app_status: ApplicationStatusOverview = serve_status.applications[app_name]
            if app_status.status == "RUNNING":
                print(f"Application '{app_name}' is RUNNING.")
                return True
            else:
                print(f"Application '{app_name}' status: {app_status.status}")
                return False
        else:
            print(f"Application '{app_name}' not found in Serve status.")
            return False
    except Exception as e:
        print(f"Error checking app status: {e}")
        return False


def run_benchmark(
    dataset: ray.data.Dataset,
    stage_type: str,
    processor_builder,
    batch_size: int,
    concurrency: int,
    **kwargs,
) -> BenchmarkResult:
    print(
        f"Benchmarking {stage_type} with batch_size={batch_size}, concurrency={concurrency}"
    )

    try:
        processor = processor_builder(batch_size, concurrency, **kwargs)

        # Execute benchmark
        start_time = time.perf_counter()
        result_dataset = processor(dataset)
        results = result_dataset.take_all()
        end_time = time.perf_counter()

        # Calculate metrics
        total_time = end_time - start_time
        total_samples = len(results)
        throughput = total_samples / total_time if total_time > 0 else 0

        print(f"{stage_type} completed: {total_samples} samples in {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} samples/s")

        return BenchmarkResult(
            stage_type=stage_type,
            batch_size=batch_size,
            concurrency=concurrency,
            total_time=total_time,
            throughput=throughput,
            total_samples=total_samples,
        )

    except Exception as e:
        raise RuntimeError(f"Error in {stage_type} benchmark: {e}")


def build_serve_deployment_processor(
    batch_size: int, concurrency: int, deployment_name: str, app_name: str
):
    """Build ServeDeployment processor for single-stage benchmark."""
    return build_llm_processor(
        config=ServeDeploymentProcessorConfig(
            deployment_name=deployment_name,
            app_name=app_name,
            dtype_mapping={
                "CompletionRequest": CompletionRequest,
                "ChatCompletionRequest": ChatCompletionRequest,
            },
            batch_size=batch_size,
            concurrency=concurrency,
        ),
        preprocess=lambda row: dict(
            method="completions",
            dtype="CompletionRequest",
            request_kwargs=dict(
                model="facebook/opt-1.3b",
                prompt=row["prompt"][:TRUNCATE_PROMPT],
                **row["sampling_params"],
            ),
        ),
        postprocess=lambda row: row,
    )


def build_shared_serve_deployment_processor(
    batch_size: int, concurrency: int, deployment_name: str, app_name: str
):
    """Build ServeDeployment processor for two-stage benchmark."""
    processor1 = build_llm_processor(
        config=ServeDeploymentProcessorConfig(
            deployment_name=deployment_name,
            app_name=app_name,
            dtype_mapping={
                "CompletionRequest": CompletionRequest,
                "ChatCompletionRequest": ChatCompletionRequest,
            },
            batch_size=batch_size,
            concurrency=concurrency,
        ),
        preprocess=lambda row: dict(
            method="completions",
            dtype="CompletionRequest",
            request_kwargs=dict(
                model="facebook/opt-1.3b",
                prompt=row["prompt"][:TRUNCATE_PROMPT],
                stream=False,
            ),
        ),
        postprocess=lambda row: dict(
            prompt=row["choices"][0]["text"] if row["choices"] else row["prompt"],
        ),
    )

    processor2 = build_llm_processor(
        config=ServeDeploymentProcessorConfig(
            deployment_name=deployment_name,
            app_name=app_name,
            dtype_mapping={
                "CompletionRequest": CompletionRequest,
                "ChatCompletionRequest": ChatCompletionRequest,
            },
            batch_size=batch_size,
            concurrency=concurrency,
        ),
        preprocess=lambda row: dict(
            method="completions",
            dtype="CompletionRequest",
            request_kwargs=dict(
                model="facebook/opt-1.3b",
                prompt=row["prompt"][:TRUNCATE_PROMPT],
                stream=False,
            ),
        ),
        postprocess=lambda row: row,
    )

    def composite_processor(dataset):
        return processor2(processor1(dataset))

    return composite_processor


def build_single_vllm_engine_processor(batch_size: int, concurrency: int):
    """Build vLLM engine processor for single-stage benchmark."""
    from ray.llm._internal.batch.processor.vllm_engine_proc import (
        build_vllm_engine_processor,
    )

    return build_vllm_engine_processor(
        config=vLLMEngineProcessorConfig(
            model_source="facebook/opt-1.3b",
            accelerator_type="A10G",
            batch_size=batch_size,
            concurrency=concurrency,
            # Disable ChatTemplate, Tokenize, Detokenize to be consistent with serve deployment benchmarks
            apply_chat_template=False,
            tokenize=False,
            detokenize=False,
            engine_kwargs=dict(
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=4096,
            ),
        ),
        preprocess=lambda row: dict(
            prompt=row["prompt"][:TRUNCATE_PROMPT],
            sampling_params=row["sampling_params"],
        ),
        postprocess=lambda row: row,
    )


def build_shared_vllm_engine_processor(batch_size: int, concurrency: int):
    """Build vLLM engine processor for two-stage benchmark."""
    from ray.llm._internal.batch.processor.vllm_engine_proc import (
        build_vllm_engine_processor,
    )

    processor1 = build_vllm_engine_processor(
        config=vLLMEngineProcessorConfig(
            model_source="facebook/opt-1.3b",
            accelerator_type="A10G",
            batch_size=batch_size,
            concurrency=concurrency,
            # Disable ChatTemplate, Tokenize, Detokenize to be consistent with serve deployment benchmarks
            apply_chat_template=False,
            tokenize=False,
            detokenize=False,
            engine_kwargs=dict(
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=4096,
            ),
        ),
        preprocess=lambda row: dict(
            prompt=row["prompt"][:TRUNCATE_PROMPT],
            sampling_params=row["sampling_params"],
        ),
        postprocess=lambda row: {
            "prompt": row["generated_text"]
            if str(row.get("generated_text", "")).strip()
            else row["prompt"]
        },
    )

    processor2 = build_vllm_engine_processor(
        config=vLLMEngineProcessorConfig(
            model_source="facebook/opt-1.3b",
            accelerator_type="A10G",
            batch_size=batch_size,
            concurrency=concurrency,
            # Disable ChatTemplate, Tokenize, Detokenize to be consistent with serve deployment benchmarks
            apply_chat_template=False,
            tokenize=False,
            detokenize=False,
            engine_kwargs=dict(
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                max_num_batched_tokens=4096,
            ),
        ),
        preprocess=lambda row: dict(
            prompt=row["prompt"][:TRUNCATE_PROMPT], sampling_params=SAMPLING_PARAMS
        ),
        postprocess=lambda row: row,
    )

    def composite_processor(dataset):
        return processor2(processor1(dataset))

    return composite_processor


def benchmark_serve_deployment(
    dataset: ray.data.Dataset,
    deployment_name: str,
    app_name: str,
    batch_size: int,
    concurrency: int,
) -> BenchmarkResult:
    return run_benchmark(
        dataset=dataset,
        stage_type="ServeDeploymentStage",
        processor_builder=build_serve_deployment_processor,
        batch_size=batch_size,
        concurrency=concurrency,
        deployment_name=deployment_name,
        app_name=app_name,
    )


def benchmark_shared_serve_deployment(
    dataset: ray.data.Dataset,
    deployment_name: str,
    app_name: str,
    batch_size: int,
    concurrency: int,
) -> BenchmarkResult:
    return run_benchmark(
        dataset=dataset,
        stage_type="SharedServeDeploymentStage",
        processor_builder=build_shared_serve_deployment_processor,
        batch_size=batch_size,
        concurrency=concurrency,
        deployment_name=deployment_name,
        app_name=app_name,
    )


def benchmark_vllm_engine(
    dataset: ray.data.Dataset, batch_size: int, concurrency: int
) -> BenchmarkResult:
    return run_benchmark(
        dataset=dataset,
        stage_type="VLLMEngineStage",
        processor_builder=build_single_vllm_engine_processor,
        batch_size=batch_size,
        concurrency=concurrency,
    )


def benchmark_shared_vllm_engine(
    dataset: ray.data.Dataset, batch_size: int, concurrency: int
) -> BenchmarkResult:
    return run_benchmark(
        dataset=dataset,
        stage_type="SharedVLLMEngineStage",
        processor_builder=build_shared_vllm_engine_processor,
        batch_size=batch_size,
        concurrency=concurrency,
    )


def print_metrics(result: BenchmarkResult):
    """Print benchmark metrics in a formatted way."""
    print("\n" + "=" * 50)
    print(f"BENCHMARK RESULTS: {result.stage_type}")
    print("=" * 50)
    print("Configuration:")
    print(f"  Batch Size: {result.batch_size}")
    print(f"  Concurrency: {result.concurrency}")
    print(f"  Total Samples: {result.total_samples}")
    print("\nPerformance Metrics:")
    print(f"  Total Time: {result.total_time:.2f} seconds")
    print(f"  Throughput: {result.throughput:.2f} samples/second")
    print("=" * 50)


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Benchmark LLM processors")
    parser.add_argument(
        "--mode",
        choices=[
            "serve_deployment",
            "vllm_engine",
            "shared_serve_deployment",
            "shared_vllm_engine",
        ],
        required=True,
        help="Which processor to benchmark",
    )

    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for processing"
    )
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrency level")

    args = parser.parse_args()

    ray.init()

    data = load_sharegpt_dataset()
    dataset = ray.data.from_items(data)

    print(f"Dataset loaded with {len(data)} samples")

    try:
        if args.mode == "serve_deployment":
            deployment_name, app_name = setup_serve_deployment(args)

            result = benchmark_serve_deployment(
                dataset, deployment_name, app_name, args.batch_size, args.concurrency
            )
            serve.delete(app_name)

        elif args.mode == "vllm_engine":
            result = benchmark_vllm_engine(dataset, args.batch_size, args.concurrency)
        elif args.mode == "shared_serve_deployment":
            deployment_name, app_name = setup_serve_deployment(args)

            result = benchmark_shared_serve_deployment(
                dataset, deployment_name, app_name, args.batch_size, args.concurrency
            )
            serve.delete(app_name)
        elif args.mode == "shared_vllm_engine":
            result = benchmark_shared_vllm_engine(
                dataset, args.batch_size, args.concurrency
            )

        print_metrics(result)

    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
