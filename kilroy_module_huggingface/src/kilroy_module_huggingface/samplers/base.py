from abc import ABC, abstractmethod

from kilroy_module_pytorch_py_sdk import Sampler as TensorSampler
from kilroy_module_server_py_sdk import Categorizable, classproperty, normalize


class Sampler(Categorizable, ABC):
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("Sampler"))

    @abstractmethod
    async def get(self) -> TensorSampler:
        pass
