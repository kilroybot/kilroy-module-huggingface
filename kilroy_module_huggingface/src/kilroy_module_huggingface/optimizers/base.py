from abc import ABC, abstractmethod
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    TypeVar,
    Union,
)

from humps import decamelize
from kilroy_module_server_py_sdk import (
    Categorizable,
    Parameter,
    classproperty,
    normalize,
)
from torch import Tensor
from torch.optim import Optimizer as TorchOptimizer

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
            # noinspection PyShadowingNames
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
    async def step(self) -> None:
        pass
