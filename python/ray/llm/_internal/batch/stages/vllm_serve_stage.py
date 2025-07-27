"""vLLM Serve Stage for shared pool configurations."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import ray
from ray.data.block import Block
from ray.data._internal.compute import TaskPoolStrategy
from ray.serve import get_app_handle

from ray.llm._internal.batch.stages.base import StatefulStage, StatefulStageUDF
from ray.llm._internal.batch.stages.vllm_engine_stage import (
    vLLMEngineRequest,
    vLLMOutputData,
    vLLMTaskType,
)
from ray.llm._internal.serve.configs.openai_api_models import (
    ChatCompletionRequest,
    CompletionRequest,
    EmbeddingRequest,
    ErrorResponse,
)

import uuid
import time

logger = logging.getLogger(__name__)


class vLLMServeStageUDF(StatefulStageUDF):
    """UDF for vLLM Serve stage that communicates with Ray Serve deployment."""

    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        batch_size: int,
        max_concurrent_batches: int,
        serve_deployment_name: str,
        task_type: vLLMTaskType = vLLMTaskType.GENERATE,
        model_id: str = None,  # Add model_id parameter
    ):
        """Initialize the vLLM Serve stage UDF.

        Args:
            data_column: The column name containing the data.
            expected_input_keys: Expected input keys for the stage.
            batch_size: The batch size for processing.
            max_concurrent_batches: Maximum number of concurrent batches.
            serve_deployment_name: Name of the Ray Serve deployment.
            task_type: The task type (generate, embed, etc.).
            model_id: The actual model ID to use in Serve requests.
        """
        super().__init__(data_column, expected_input_keys)
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.serve_deployment_name = serve_deployment_name
        self.task_type = task_type
        self.model_id = model_id  # Store the model ID
        self._serve_handle = None

    async def _get_serve_handle(self):
        """Get the Ray Serve handle for the deployment with streaming enabled."""
        if self._serve_handle is None:
            try:
                base_handle = get_app_handle(self.serve_deployment_name)
                # Enable streaming mode so that the deployment can legally return async generators.
                self._serve_handle = base_handle.options(stream=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get serve handle for {self.serve_deployment_name}"
                )
        return self._serve_handle

    async def _process_request(self, request: vLLMEngineRequest) -> vLLMOutputData:
        """Process a single request through the Serve deployment."""
        serve_handle = await self._get_serve_handle()

        # Convert vLLM request to appropriate Serve request format
        if self.task_type == vLLMTaskType.GENERATE:
            serve_request = CompletionRequest(
                model=self.model_id,
                prompt=request.prompt,
                stream=False,
            )

            response_gen = serve_handle.completions.remote(serve_request)
            response = await response_gen.__anext__()

            if isinstance(response, ErrorResponse):
                raise RuntimeError(f"Serve deployment error: {response.message}")

            # TODO (ycwwang): Can we use vLLMOutputData.from_vllm_engine_output?
            output_data = vLLMOutputData(
                prompt=request.prompt,
                prompt_token_ids=request.prompt_token_ids,
                num_input_tokens=len(request.prompt_token_ids)
                if request.prompt_token_ids
                else 0,
                generated_text=response.choices[0].text,
                num_generated_tokens=len(response.choices[0].text.split()),
            )

        elif self.task_type == vLLMTaskType.EMBED:
            serve_request = EmbeddingRequest(
                model=self.model_id,
                input=request.prompt,
            )

            response_gen = serve_handle.embeddings.remote(serve_request)
            response = await response_gen.__anext__()

            if isinstance(response, ErrorResponse):
                raise RuntimeError(f"Serve deployment error: {response.message}")

            output_data = vLLMOutputData(
                prompt=request.prompt,
                prompt_token_ids=request.prompt_token_ids,
                num_input_tokens=len(request.prompt_token_ids)
                if request.prompt_token_ids
                else 0,
                embeddings=response.data[0].embedding,
            )

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return output_data

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """Process a batch of requests through the Serve deployment."""
        # The batch is already a list of dictionaries from StatefulStageUDF.__call__
        requests = []

        for row in batch:
            # Convert dict to vLLMEngineRequest if needed
            if isinstance(row, vLLMEngineRequest):
                requests.append(row)
            else:
                # Create a vLLMEngineRequest from the row data
                # Extract required fields
                prompt = row.get("prompt")
                if not prompt:
                    raise ValueError("Missing required field 'prompt' in row")

                # Extract sampling parameters
                sampling_params = row.get("sampling_params", {})

                # Create a simple vLLMEngineRequest
                request = vLLMEngineRequest(
                    request_id=0,  # Will be set by the engine
                    idx_in_batch=row.get(self.IDX_IN_BATCH_COLUMN, 0),
                    prompt=prompt,
                    prompt_token_ids=row.get("prompt_token_ids"),
                    images=row.get("images", []),
                    params=sampling_params,  # This will be converted by the Serve deployment
                    lora_request=None,
                )
                requests.append(request)

        # Process requests concurrently using as_completed pattern like vLLMEngineStageUDF
        batch_uuid = uuid.uuid4()
        t = time.perf_counter()

        # Create tasks with request info for proper indexing
        tasks = []
        for i, req in enumerate(requests):
            # Create a wrapper that includes the index
            async def process_with_index(request, idx):
                result = await self._process_request(request)
                return idx, result

            task = asyncio.create_task(process_with_index(req, i))
            tasks.append(task)

        time_taken = -1.0
        for completed_task in asyncio.as_completed(tasks):
            req_idx, result = await completed_task
            time_taken = time.perf_counter() - t

            # TODO (ycwwang): Inspect what is in result
            # Match the output format of vLLMEngineStageUDF
            output_dict = {
                "generated_text": result.generated_text,
                "generated_tokens": result.generated_tokens,
                "num_generated_tokens": result.num_generated_tokens,
                "num_input_tokens": result.num_input_tokens,
                "request_id": 0,  # We don't have request_id from Serve, use 0
                self.IDX_IN_BATCH_COLUMN: req_idx,
                "batch_uuid": batch_uuid.hex,
                "time_taken_llm": time_taken,
                "params": str(result.num_input_tokens),  # Use a placeholder for params
            }

            # Add embeddings if present
            if result.embeddings is not None:
                output_dict["embeddings"] = result.embeddings

            yield output_dict

        logger.info(
            "[vLLM Serve] Elapsed time for batch %s with size %d: %s",
            batch_uuid.hex,
            len(batch),
            time_taken,
        )


class vLLMServeStage(StatefulStage):
    """Stage for vLLM processing using Ray Serve deployment."""

    fn: type[vLLMServeStageUDF] = vLLMServeStageUDF

    def __init__(
        self,
        serve_deployment_name: str,
        task_type: vLLMTaskType = vLLMTaskType.GENERATE,
        batch_size: int = 32,
        max_concurrent_batches: int = 8,
        concurrency: int = 1,
        model_id: str = None,  # Add model_id parameter
        **kwargs,
    ):
        """Initialize the vLLM Serve stage.

        Args:
            serve_deployment_name: Name of the Ray Serve deployment.
            task_type: The task type (generate, embed, etc.).
            batch_size: The batch size for processing.
            max_concurrent_batches: Maximum number of concurrent batches.
            concurrency: The number of concurrent tasks to use.
            model_id: The actual model ID to use in Serve requests.
            **kwargs: Additional arguments for map_batches.
        """
        fn_constructor_kwargs = {
            "serve_deployment_name": serve_deployment_name,
            "task_type": task_type,
            "batch_size": batch_size,
            "max_concurrent_batches": max_concurrent_batches,
            "model_id": model_id,  # Pass model_id to UDF
        }

        # Use TaskPoolStrategy with concurrency from shared pool config
        map_batches_kwargs = {
            "zero_copy_batch": True,
            "concurrency": concurrency,
            "batch_size": batch_size,
            **kwargs,
        }

        super().__init__(
            fn=vLLMServeStageUDF,
            fn_constructor_kwargs=fn_constructor_kwargs,
            map_batches_kwargs=map_batches_kwargs,
        )
