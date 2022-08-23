from dataclasses import dataclass
from typing import Any, Dict

from kilroy_module_server_py_sdk import (
    Configurable,
    SerializableModel,
    background,
    classproperty,
)
from torch.optim import RMSprop

from kilroy_module_huggingface.optimizers.base import (
    Optimizer,
    OptimizerParameter,
)


class Params(SerializableModel):
    lr: float = 0.001
    momentum: float = 0
    alpha: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0


@dataclass
class State:
    optimizer: RMSprop


class RMSPropOptimizer(Optimizer[RMSprop], Configurable[State]):
    class LrParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class MomentumParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class AlphaParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class EpsParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    class WeightDecayParameter(OptimizerParameter[State, float]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {"type": "number", "minimum": 0}

    async def build_default_state(self) -> State:
        user_params = Params(**self._kwargs)
        return State(optimizer=RMSprop(self._params, **user_params.dict()))

    async def step(self) -> None:
        async with self.state.write_lock() as state:

            def step():
                state.optimizer.step()
                state.optimizer.zero_grad()

            await background(step)
