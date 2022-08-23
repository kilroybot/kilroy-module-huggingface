from abc import ABC, abstractmethod
from typing import AsyncIterable, Set, Tuple

from kilroy_module_server_py_sdk import (
    Categorizable,
    Metric,
    classproperty,
    normalize,
)
from torch import Tensor

from kilroy_module_huggingface.models import HuggingfaceLanguageModel
from kilroy_module_huggingface.optimizers import Optimizer


class Trainer(Categorizable, ABC):
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Trainer"))

    @abstractmethod
    async def fit_supervised(
        self, model: HuggingfaceLanguageModel, data: AsyncIterable[Tensor]
    ) -> None:
        pass

    @abstractmethod
    async def fit_reinforced(
        self,
        model: HuggingfaceLanguageModel,
        results: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        pass

    @abstractmethod
    async def step(self, optimizer: Optimizer) -> None:
        pass

    async def get_metrics(self) -> Set[Metric]:
        return set()
