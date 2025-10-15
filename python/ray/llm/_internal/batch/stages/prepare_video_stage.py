from typing import Any, AsyncIterator, Dict, List

from ray.llm._internal.batch.stages.base import StatefulStage, StatefulStageUDF
from qwen_vl_utils import process_vision_info


class PrepareVideoUDF(StatefulStageUDF):
    """Extract video information from OpenAI chat messages.

    This UDF follows the contract defined in ``StatefulStageUDF``:
    it takes a batch of rows (each row is a dict) where the ``messages`` key
    contains a list of chat messages in the OpenAI format.  It extracts the
    video inputs referenced in each request and yields a row that includes
    (1) ``__idx_in_batch`` so that downstream stages can align the outputs and
    (2) a ``video`` field that holds the list of parsed video objects/URLs.

    Unlike image preparation, we *don't* download the video bytes – we just
    forward the raw video inputs (e.g., a URL string or model-specific video
    representation) because vLLM's vision-language models expect that format
    directly.  This keeps the implementation minimal for a proof-of-concept.
    """

    def __init__(self, data_column: str, expected_input_keys: List[str]):
        super().__init__(data_column, expected_input_keys)

    async def udf(
        self, batch: List[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:  # noqa: D401 – keep signature consistent
        # Each element in ``batch`` is the row produced by previous stages.
        # ``messages`` is required (validated by previous stages).

        # Extract per-request video inputs.
        all_video_inputs: List[List[Any]] = []
        for row in batch:
            # ``process_vision_info`` returns (image_inputs, video_inputs).
            _, video_inputs = process_vision_info(row["messages"])
            all_video_inputs.append(video_inputs or [])

        # Emit outputs keeping ordering via __idx_in_batch.
        for idx, video_inputs in enumerate(all_video_inputs):
            ret: Dict[str, Any] = {self.IDX_IN_BATCH_COLUMN: idx}
            if video_inputs:
                ret["video"] = video_inputs
            yield ret


class PrepareVideoStage(StatefulStage):
    """A stage that prepares video inputs from chat template messages."""

    fn: StatefulStageUDF = PrepareVideoUDF

    def get_required_input_keys(self) -> Dict[str, str]:
        return {
            "messages": "A list of messages in OpenAI chat format. "
            "See https://platform.openai.com/docs/api-reference/chat/create "
            "for details."
        }