import os
from typing import Dict, List, Optional, Union

from .common import NodeIdStr
from ray.data._internal.execution.util import memory_string
from ray.util.annotations import DeveloperAPI


class ExecutionResources:
    """Specifies resources usage or resource limits for execution.

    By default this class represents resource usage. Use `for_limits` or
    set `default_to_inf` to True to create an object that represents resource limits.
    """

    def __init__(
        self,
        cpu: Optional[float] = None,
        gpu: Optional[float] = None,
        object_store_memory: Optional[float] = None,
        memory: Optional[float] = None,
        default_to_inf: bool = False,
    ):
        """Initializes ExecutionResources.
        Args:
            cpu: Amount of logical CPU slots.
            gpu: Amount of logical GPU slots.
            object_store_memory: Amount of object store memory.
            memory: Amount of logical memory in bytes.
            default_to_inf: When the object represents resource usage, this flag
                should be set to False. And missing values will default to 0.
                When the object represents resource limits, this flag should be
                set to True. And missing values will default to infinity.
        """
        self._cpu = cpu
        self._gpu = gpu
        self._object_store_memory = object_store_memory
        self._memory = memory
        self._default_to_inf = default_to_inf

    @classmethod
    def from_resource_dict(
        cls,
        resource_dict: Dict[str, float],
        default_to_inf: bool = False,
    ):
        """Create an ExecutionResources object from a resource dict."""
        return ExecutionResources(
            cpu=resource_dict.get("CPU", None) or resource_dict.get("num_cpus", None),
            gpu=resource_dict.get("GPU", None) or resource_dict.get("num_gpus", None),
            object_store_memory=resource_dict.get("object_store_memory", None),
            memory=resource_dict.get("memory", None),
            default_to_inf=default_to_inf,
        )

    @classmethod
    def for_limits(
        cls,
        cpu: Optional[float] = None,
        gpu: Optional[float] = None,
        object_store_memory: Optional[float] = None,
        memory: Optional[float] = None,
    ) -> "ExecutionResources":
        """Create an ExecutionResources object that represents resource limits.
        Args:
            cpu: Amount of logical CPU slots.
            gpu: Amount of logical GPU slots.
            object_store_memory: Amount of object store memory.
            memory: Amount of logical memory in bytes.
        """
        return ExecutionResources(
            cpu=cpu,
            gpu=gpu,
            object_store_memory=object_store_memory,
            memory=memory,
            default_to_inf=True,
        )

    @property
    def cpu(self) -> float:
        if self._cpu is not None:
            return self._cpu
        return 0.0 if not self._default_to_inf else float("inf")

    @cpu.setter
    def cpu(self, value: float):
        self._cpu = value

    @property
    def gpu(self) -> float:
        if self._gpu is not None:
            return self._gpu
        return 0.0 if not self._default_to_inf else float("inf")

    @gpu.setter
    def gpu(self, value: float):
        self._gpu = value

    @property
    def object_store_memory(self) -> float:
        if self._object_store_memory is not None:
            return self._object_store_memory
        return 0.0 if not self._default_to_inf else float("inf")

    @object_store_memory.setter
    def object_store_memory(self, value: float):
        self._object_store_memory = value

    @property
    def memory(self) -> float:
        if self._memory is not None:
            return self._memory
        return 0.0 if not self._default_to_inf else float("inf")

    @memory.setter
    def memory(self, value: float):
        self._memory = value

    def __repr__(self):
        return (
            f"ExecutionResources(cpu={self.cpu:.1f}, gpu={self.gpu:.1f}, "
            f"object_store_memory={self.object_store_memory_str()}, "
            f"memory={self.memory_str()})"
        )

    def __eq__(self, other: "ExecutionResources") -> bool:
        return (
            self.cpu == other.cpu
            and self.gpu == other.gpu
            and self.object_store_memory == other.object_store_memory
            and self.memory == other.memory
        )

    @classmethod
    def zero(cls) -> "ExecutionResources":
        """Returns an ExecutionResources object with zero resources."""
        return ExecutionResources(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def inf(cls) -> "ExecutionResources":
        """Returns an ExecutionResources object with infinite resources."""
        return ExecutionResources.for_limits()

    def is_zero(self) -> bool:
        """Returns True if all resources are zero."""
        return (
            self.cpu == 0.0
            and self.gpu == 0.0
            and self.object_store_memory == 0.0
            and self.memory == 0.0
        )

    def is_non_negative(self) -> bool:
        """Returns True if all resources are non-negative."""
        return (
            self.cpu >= 0
            and self.gpu >= 0
            and self.object_store_memory >= 0
            and self.memory >= 0
        )

    def object_store_memory_str(self) -> str:
        """Returns a human-readable string for the object store memory field."""
        if self.object_store_memory == float("inf"):
            return "inf"
        return memory_string(self.object_store_memory)

    def memory_str(self) -> str:
        """Returns a human-readable string for the memory field."""
        if self.memory == float("inf"):
            return "inf"
        return memory_string(self.memory)

    def copy(self) -> "ExecutionResources":
        """Returns a copy of this ExecutionResources object."""
        return ExecutionResources(
            cpu=self._cpu,
            gpu=self._gpu,
            object_store_memory=self._object_store_memory,
            memory=self._memory,
            default_to_inf=self._default_to_inf,
        )

    def add(self, other: "ExecutionResources") -> "ExecutionResources":
        """Adds execution resources.

        Returns:
            A new ExecutionResource object with summed resources.
        """
        return ExecutionResources(
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
            object_store_memory=self.object_store_memory + other.object_store_memory,
            memory=self.memory + other.memory,
        )

    def subtract(self, other: "ExecutionResources") -> "ExecutionResources":
        """Subtracts execution resources.

        Returns:
            A new ExecutionResource object with subtracted resources.
        """
        return ExecutionResources(
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
            object_store_memory=self.object_store_memory - other.object_store_memory,
            memory=self.memory - other.memory,
        )

    def max(self, other: "ExecutionResources") -> "ExecutionResources":
        """Returns the maximum for each resource type."""
        return ExecutionResources(
            cpu=max(self.cpu, other.cpu),
            gpu=max(self.gpu, other.gpu),
            object_store_memory=max(
                self.object_store_memory, other.object_store_memory
            ),
            memory=max(self.memory, other.memory),
        )

    def min(self, other: "ExecutionResources") -> "ExecutionResources":
        """Returns the minimum for each resource type."""
        return ExecutionResources(
            cpu=min(self.cpu, other.cpu),
            gpu=min(self.gpu, other.gpu),
            object_store_memory=min(
                self.object_store_memory, other.object_store_memory
            ),
            memory=min(self.memory, other.memory),
        )

    def satisfies_limit(
        self,
        limit: "ExecutionResources",
        *,
        ignore_object_store_memory=False,
    ) -> bool:
        """Return if this resource struct meets the specified limits.

        Note that None for a field means no limit.

        Args:
            limit: The resource limits to check against.
            ignore_object_store_memory: If True, ignore the object store memory
                limit when checking if this resource struct meets the limits.
        """
        return (
            self.cpu <= limit.cpu
            and self.gpu <= limit.gpu
            and (
                ignore_object_store_memory
                or self.object_store_memory <= limit.object_store_memory
            )
            and self.memory <= limit.memory
        )

    def scale(self, f: float) -> "ExecutionResources":
        """Return copy with all set values scaled by `f`."""
        if f < 0:
            raise ValueError("Scaling factor must be non-negative.")
        if f == 0:
            # Explicitly handle the zero case, because `0 * inf` is undefined.
            return ExecutionResources.zero()
        return ExecutionResources(
            cpu=self.cpu * f,
            gpu=self.gpu * f,
            object_store_memory=self.object_store_memory * f,
            memory=self.memory * f,
        )


@DeveloperAPI
class ExecutionOptions:
    """Common options for execution.

    Some options may not be supported on all executors (e.g., resource limits).

    Attributes:
        resource_limits: Set a soft limit on the resource usage during execution.
            Autodetected by default.
        exclude_resources: Amount of resources to exclude from Ray Data.
            Set this if you have other workloads running on the same cluster.
            Note,
            - If using Ray Data with Ray Train, training resources will be
            automatically excluded.
            - For each resource type, resource_limits and exclude_resources can
            not be both set.
        locality_with_output: Set this to prefer running tasks on the same node as the
            output node (node driving the execution). It can also be set to a list of
            node ids to spread the outputs across those nodes. Off by default.
        preserve_order: Set this to preserve the ordering between blocks processed by
            operators. Off by default.
        actor_locality_enabled: Whether to enable locality-aware task dispatch to
            actors (off by default). This parameter applies to both stateful map and
            streaming_split operations.
        verbose_progress: Whether to report progress individually per operator. By
            default, only AllToAll operators and global progress is reported. This
            option is useful for performance debugging. On by default.
    """

    def __init__(
        self,
        resource_limits: Optional[ExecutionResources] = None,
        exclude_resources: Optional[ExecutionResources] = None,
        locality_with_output: Union[bool, List[NodeIdStr]] = False,
        preserve_order: bool = False,
        actor_locality_enabled: bool = True,
        verbose_progress: Optional[bool] = None,
    ):
        if resource_limits is None:
            resource_limits = ExecutionResources.for_limits()
        self.resource_limits = resource_limits
        if exclude_resources is None:
            exclude_resources = ExecutionResources.zero()
        self.exclude_resources = exclude_resources
        self.locality_with_output = locality_with_output
        self.preserve_order = preserve_order
        self.actor_locality_enabled = actor_locality_enabled
        if verbose_progress is None:
            verbose_progress = bool(
                int(os.environ.get("RAY_DATA_VERBOSE_PROGRESS", "1"))
            )
        self.verbose_progress = verbose_progress

    def __repr__(self) -> str:
        return (
            f"ExecutionOptions(resource_limits={self.resource_limits}, "
            f"exclude_resources={self.exclude_resources}, "
            f"locality_with_output={self.locality_with_output}, "
            f"preserve_order={self.preserve_order}, "
            f"actor_locality_enabled={self.actor_locality_enabled}, "
            f"verbose_progress={self.verbose_progress})"
        )

    @property
    def resource_limits(self) -> ExecutionResources:
        return self._resource_limits

    @resource_limits.setter
    def resource_limits(self, value: ExecutionResources) -> None:
        self._resource_limits = ExecutionResources.for_limits(
            cpu=value._cpu,
            gpu=value._gpu,
            object_store_memory=value._object_store_memory,
            memory=value._memory,
        )

    def is_resource_limits_default(self):
        """Returns True if resource_limits is the default value."""
        return self._resource_limits == ExecutionResources.for_limits()

    def validate(self) -> None:
        """Validate the options."""
        for attr in ["cpu", "gpu", "object_store_memory"]:
            if (
                getattr(self.resource_limits, attr) != float("inf")
                and getattr(self.exclude_resources, attr, 0) > 0
            ):
                raise ValueError(
                    "resource_limits and exclude_resources cannot "
                    f" both be set for {attr} resource."
                )
