from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    TypeVar,
    Union,
)

from humps import decamelize
from kilroy_module_server_py_sdk import (
    Categorizable,
    Configurable,
    Parameter,
    SerializableModel,
    classproperty,
    normalize,
)
from torch import Tensor
from torch.optim import Adam, Optimizer as TorchOptimizer, RMSprop, SGD

StateType = TypeVar("StateType")
ParameterType = TypeVar("ParameterType")
OptimizerType = TypeVar("OptimizerType", bound=TorchOptimizer)


class OptimizerParameter(
    Parameter[StateType, ParameterType], Generic[StateType, ParameterType], ABC
):
    def _get_param(self, group: Dict[str, Any]) -> ParameterType:
        return group[decamelize(self.name)]

    def _set_param(self, group: Dict[str, Any], value: ParameterType) -> None:
        group[decamelize(self.name)] = value

    async def _get(self, state: StateType) -> ParameterType:
        return self._get_param(state.optimizer.param_groups[0])

    async def _set(
        self, state: StateType, value: ParameterType
    ) -> Callable[[], Awaitable]:
        original_value = self._get_param(state.optimizer.param_groups[0])

        async def undo() -> None:
            for param_group in state.optimizer.param_groups:
                self._set_param(param_group, original_value)

        for param_group in state.optimizer.param_groups:
            self._set_param(param_group, value)

        return undo


class Optimizer(Categorizable, Generic[OptimizerType], ABC):
    def __init__(
        self, params: Union[Iterable[Tensor], Iterable[Dict]], *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._params = list(params)

    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Optimizer"))

    @abstractmethod
    async def get(self) -> OptimizerType:
        pass


# Adam


class AdamOptimizerParams(SerializableModel):
    lr: float = 0.001
    betas: List[float] = [0.9, 0.999]
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class AdamOptimizerState:
    optimizer: Adam


class AdamOptimizer(Optimizer[Adam], Configurable[AdamOptimizerState]):
    class LrParameter(OptimizerParameter[AdamOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class Beta1Parameter(OptimizerParameter[AdamOptimizerState, float]):
        def _get_param(self, group: Dict[str, Any]) -> ParameterType:
            return group["betas"][0]

        def _set_param(
            self, group: Dict[str, Any], value: ParameterType
        ) -> None:
            group["betas"][0] = value

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class Beta2Parameter(OptimizerParameter[AdamOptimizerState, float]):
        def _get_param(self, group: Dict[str, Any]) -> ParameterType:
            return group["betas"][1]

        def _set_param(
            self, group: Dict[str, Any], value: ParameterType
        ) -> None:
            group["betas"][1] = value

        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class EpsParameter(OptimizerParameter[AdamOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(OptimizerParameter[AdamOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def build_default_state(self) -> AdamOptimizerState:
        user_params = AdamOptimizerParams(**self._kwargs)
        return AdamOptimizerState(
            optimizer=Adam(self._params, **user_params.dict())
        )

    async def get(self) -> Adam:
        async with self.state.read_lock() as state:
            return state.optimizer


# RMSProp


class RMSpropOptimizerParams(SerializableModel):
    lr: float = 0.001
    momentum: float = 0
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class RMSPropOptimizerState:
    optimizer: RMSprop


class RMSPropOptimizer(
    Optimizer[RMSprop], Configurable[RMSPropOptimizerState]
):
    class LrParameter(OptimizerParameter[RMSPropOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class MomentumParameter(OptimizerParameter[RMSPropOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class AlphaParameter(OptimizerParameter[RMSPropOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class EpsParameter(OptimizerParameter[RMSPropOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(
        OptimizerParameter[RMSPropOptimizerState, float]
    ):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def build_default_state(self) -> RMSPropOptimizerState:
        user_params = RMSpropOptimizerParams(**self._kwargs)
        return RMSPropOptimizerState(
            optimizer=RMSprop(self._params, **user_params.dict())
        )

    async def get(self) -> RMSprop:
        async with self.state.read_lock() as state:
            return state.optimizer


# SGD


class SGDOptimizerParams(SerializableModel):
    lr: float = 0.01
    momentum: float = 0
    weight_decay: float = 0
    dampening: float = 0


@dataclass
class SGDOptimizerState:
    optimizer: SGD


class SGDOptimizer(Optimizer[SGD], Configurable[SGDOptimizerState]):
    class LrParameter(OptimizerParameter[SGDOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class MomentumParameter(OptimizerParameter[SGDOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(OptimizerParameter[SGDOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class DampeningParameter(OptimizerParameter[SGDOptimizerState, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def build_default_state(self) -> SGDOptimizerState:
        user_params = SGDOptimizerParams(**self._kwargs)
        return SGDOptimizerState(
            optimizer=SGD(self._params, **user_params.dict())
        )

    async def get(self) -> SGD:
        async with self.state.read_lock() as state:
            return state.optimizer
