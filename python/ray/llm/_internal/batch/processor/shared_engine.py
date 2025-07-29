"""Shared engine registry for managing shared LLM engine configurations."""

from collections import defaultdict
from typing import Dict, Optional

from pydantic import BaseModel

from ray.llm._internal.batch.processor.base import ProcessorConfig


class SharedEngineInfo(BaseModel):
    """
    Tracks a shared LLM engine configuration and its associated Ray Serve deployment.

    When multiple processors share the same LLM engine configuration, they can reuse
    the same vLLM engine instance via a Ray Serve deployment. This class tracks
    which deployment name corresponds to which shared engine configuration.

    Note: Only the LLM engine is shared, not other processor stages like tokenization,
    chat template, etc.
    """

    processor_config: ProcessorConfig
    deployment_name: Optional[str] = None


class _SharedEngineRegistry:
    """Registry for tracking shared LLM engine configurations and their deployments."""

    def __init__(self):
        self._shared_configs: Dict[int, SharedEngineInfo] = defaultdict(
            SharedEngineInfo
        )

    def register_config(self, config: ProcessorConfig) -> None:
        """Register a processor configuration for shared LLM engine usage."""
        config_id = id(config)
        if config_id not in self._shared_configs:
            self._shared_configs[config_id] = SharedEngineInfo(processor_config=config)

    def get_shared_info(self, config: ProcessorConfig) -> Optional[SharedEngineInfo]:
        """Get the shared engine info for a processor configuration."""
        config_id = id(config)
        return self._shared_configs.get(config_id)

    def set_serve_deployment(self, config: ProcessorConfig, deployment: str) -> None:
        """Set the Ray Serve deployment name for a shared engine configuration."""
        config_id = id(config)
        if config_id in self._shared_configs:
            self._shared_configs[config_id].deployment_name = deployment


# Global shared engine registry
_shared_engine_registry = _SharedEngineRegistry()
