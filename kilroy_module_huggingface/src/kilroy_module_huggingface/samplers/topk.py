from typing import Any, Dict

from kilroy_module_pytorch_py_sdk import (
    EpsilonTopKCategoricalSampler,
    Sampler as TensorSampler,
    TopKCategoricalSampler,
)
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
)

from kilroy_module_huggingface.samplers.base import Sampler


class State(SerializableState):
    k: int = 10


class TopKSampler(Sampler, Configurable[State]):
    class KParameter(Parameter[State, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return TopKCategoricalSampler(state.k)


# Epsilon


class EpsilonState(SerializableState):
    k: int = 10
    epsilon: float = 0.01


class EpsilonTopKSampler(Sampler, Configurable[EpsilonState]):
    class KParameter(Parameter[EpsilonState, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(self) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonTopKCategoricalSampler(
                k=state.k, epsilon=state.epsilon
            )
