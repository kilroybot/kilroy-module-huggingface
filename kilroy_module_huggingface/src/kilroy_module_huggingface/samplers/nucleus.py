from typing import Any, Dict

from kilroy_module_pytorch_py_sdk import (
    EpsilonNucleusCategoricalSampler,
    NucleusCategoricalSampler,
    Sampler as TensorSampler,
)
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
)

from kilroy_module_huggingface.samplers.base import Sampler


class State(SerializableState):
    p: float = 0.95


class NucleusSampler(Sampler, Configurable[State]):
    class PParameter(Parameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return NucleusCategoricalSampler(state.p)


# EpsilonNucleus


class EpsilonState(SerializableState):
    p: float = 0.95
    epsilon: float = 0.01


class EpsilonNucleusSampler(Sampler, Configurable[EpsilonState]):
    class PParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    class EpsilonParameter(Parameter[EpsilonState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonNucleusCategoricalSampler(
                p=state.p, epsilon=state.epsilon
            )
