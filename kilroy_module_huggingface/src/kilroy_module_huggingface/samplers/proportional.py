from typing import Any, Dict

from kilroy_module_pytorch_py_sdk import (
    EpsilonProportionalCategoricalSampler,
    ProportionalCategoricalSampler,
    Sampler as TensorSampler,
)
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
)

from kilroy_module_huggingface.samplers.base import Sampler


class ProportionalSampler(Sampler):
    async def get(self) -> TensorSampler:
        return ProportionalCategoricalSampler()


# Epsilon


class EpsilonState(SerializableState):
    epsilon: float = 0.01


class EpsilonProportionalSampler(Sampler, Configurable[EpsilonState]):
    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonProportionalCategoricalSampler(epsilon=state.epsilon)
