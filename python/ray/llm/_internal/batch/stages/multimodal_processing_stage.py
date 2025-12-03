"""Multimodal Processing Stage"""

from typing import Any, AsyncIterator, Dict, List

from ray.llm._internal.batch.stages.base import StatefulStage, StatefulStageUDF
from ray.llm._internal.batch.stages.common import maybe_convert_ndarray_to_list


class MultimodalProcessingUDF(StatefulStageUDF):
    def __init__(
        self,
        data_column: str,
        expected_input_keys: List[str],
        model: str,
    ):
        """
        Initialize the MultimodalProcessingUDF.

        Args:
            data_column: The data column name.
            expected_input_keys: The expected input keys of the stage.
            model: The model to use for the multimodal processor.
        """
        from vllm.config import ModelConfig, VllmConfig

        super().__init__(data_column, expected_input_keys)
        self.model_config = ModelConfig(model=model)
        self.vllm_config = VllmConfig(
            model_config=self.model_config,
        )

        from vllm.transformers_utils.tokenizer import init_tokenizer_from_configs
        tokenizer = init_tokenizer_from_configs(self.model_config)

        from vllm.v1.engine.processor import Processor
        self.processor = Processor(self.vllm_config, tokenizer)
        self.request_id = 0

    async def udf(self, batch: List[Dict[str, Any]]) -> AsyncIterator[Dict[str, Any]]:
        """
        Process multimodal data and create engine_core_request.

        Args:
            batch: A list of rows to process.

        Yields:
            Dict[str, Any]: A dictionary containing the engine_core_request.
        """
        import vllm

        for row in batch:
            prompt = row.get("prompt", None)
            tokenized_prompt = row.get("tokenized_prompt", None)
            if tokenized_prompt is not None:
                tokenized_prompt = maybe_convert_ndarray_to_list(tokenized_prompt)

            multimodal_data = row.get("multimodal_data", None)
            mm_processor_kwargs = row.get("mm_processor_kwargs", None)
            multimodal_uuids = row.get("multimodal_uuids", None)

            multi_modal_data = multimodal_data

            # Prepare sampling parameters
            sampling_params = row.get("sampling_params", None)
            if sampling_params is not None:
                sampling_params = sampling_params.copy() if isinstance(sampling_params, dict) else sampling_params
                if isinstance(sampling_params, dict) and "guided_decoding" in sampling_params:
                    guided_decoding_dict = sampling_params.pop("guided_decoding")
                    if isinstance(guided_decoding_dict, dict):
                        guided_decoding = vllm.sampling_params.GuidedDecodingParams(
                            **maybe_convert_ndarray_to_list(guided_decoding_dict)
                        )
                    else:
                        guided_decoding = None
                else:
                    guided_decoding = None
                
                params = vllm.SamplingParams(
                    **maybe_convert_ndarray_to_list(sampling_params),
                    guided_decoding=guided_decoding,
                )
            else:
                # EMBED task - use PoolingParams
                params = vllm.PoolingParams(task="embed")

            # Create llm_prompt
            if tokenized_prompt is not None:
                llm_prompt = vllm.inputs.data.TokensPrompt(
                    prompt_token_ids=tokenized_prompt,
                    multi_modal_data=multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    multi_modal_uuids=multimodal_uuids,
                )
            else:
                assert prompt, "Either prompt or tokenized_prompt must be provided"
                llm_prompt = vllm.inputs.data.TextPrompt(
                    prompt=prompt,
                    multi_modal_data=multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                    multi_modal_uuids=multimodal_uuids,
                )

            # Create engine_core_request
            from vllm.v1.engine import EngineCoreRequest
            # Use a temporary request_id (will be replaced in vllm_engine_stage)
            engine_core_request = self.processor.process_inputs(
                request_id=str(self.request_id),
                prompt=llm_prompt,
                params=params,
            )
            self.request_id += 1

            yield {
                self.IDX_IN_BATCH_COLUMN: row[self.IDX_IN_BATCH_COLUMN],
                "engine_core_request": engine_core_request,
            }

class MultimodalProcessingStage(StatefulStage):
    """
    A stage that processes multimodal data and creates engine_core_request.
    """

    fn: StatefulStageUDF = MultimodalProcessingUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        """The required input keys of the stage and their descriptions."""
        return {
            "prompt": "The text prompt (str). Either prompt or tokenized_prompt must be provided.",
        }

    def get_optional_input_keys(self) -> Dict[str, str]:
        """The optional input keys of the stage and their descriptions."""
        return {
            "tokenized_prompt": "The tokenized prompt. If provided, prompt will not be used.",
            "sampling_params": "The sampling parameters for GENERATE task.",
            "multimodal_data": "The multimodal data to pass to the model.",
            "mm_processor_kwargs": "The kwargs for the engine's multimodal processor.",
            "multimodal_uuids": "User-specified UUIDs for multimodal items, mapped by modality.",
            "image": "The image(s) for multimodal input. Accepts a single image or list of images.",
        }