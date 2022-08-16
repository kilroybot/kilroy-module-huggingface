from abc import ABC, abstractmethod
from typing import Any, Dict

from kilroy_module_pytorch_py_sdk import (
    EpsilonNucleusCategoricalSampler,
    EpsilonProportionalCategoricalSampler,
    EpsilonTopKCategoricalSampler,
    NucleusCategoricalSampler,
    ProportionalCategoricalSampler,
    Sampler as TensorSampler,
    TopKCategoricalSampler,
)
from kilroy_module_server_py_sdk import (
    Categorizable,
    Configurable,
    Parameter,
    SerializableState,
    classproperty,
    normalize,
)


class Sampler(Categorizable, ABC):
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Sampler"))

    @abstractmethod
    async def get(self) -> TensorSampler:
        pass


# Proportional


class ProportionalSampler(Sampler):
    async def get(self) -> TensorSampler:
        return ProportionalCategoricalSampler()


# TopK


class TopKSamplerState(SerializableState):
    k: int = 10


class TopKSampler(Sampler, Configurable[TopKSamplerState]):
    class KParameter(Parameter[TopKSamplerState, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return TopKCategoricalSampler(state.k)


# Nucleus


class NucleusSamplerState(SerializableState):
    p: float = 0.9


class NucleusSampler(Sampler, Configurable[NucleusSamplerState]):
    class PParameter(Parameter[NucleusSamplerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return NucleusCategoricalSampler(state.p)


# EpsilonProportional


class EpsilonProportionalSamplerState(SerializableState):
    epsilon: float = 0.1


class EpsilonProportionalSampler(
    Sampler, Configurable[EpsilonProportionalSamplerState]
):
    class EpsilonParameter(Parameter[EpsilonProportionalSamplerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonProportionalCategoricalSampler(epsilon=state.epsilon)


# EpsilonTopK


class EpsilonTopKSamplerState(SerializableState):
    k: int = 10
    epsilon: float = 0.1


class EpsilonTopKSampler(Sampler, Configurable[EpsilonTopKSamplerState]):
    class KParameter(Parameter[EpsilonTopKSamplerState, int]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "integer", "minimum": 1}

    class EpsilonParameter(Parameter[EpsilonTopKSamplerState, float]):
        @classproperty
        def schema(self) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonTopKCategoricalSampler(
                k=state.k, epsilon=state.epsilon
            )


# EpsilonNucleus


class EpsilonNucleusSamplerState(SerializableState):
    p: float = 0.9
    epsilon: float = 0.1


class EpsilonNucleusSampler(Sampler, Configurable[EpsilonNucleusSamplerState]):
    class PParameter(Parameter[EpsilonNucleusSamplerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    class EpsilonParameter(Parameter[EpsilonNucleusSamplerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0, "maximum": 1}

    async def get(self) -> TensorSampler:
        async with self.state.read_lock() as state:
            return EpsilonNucleusCategoricalSampler(
                p=state.p, epsilon=state.epsilon
            )
