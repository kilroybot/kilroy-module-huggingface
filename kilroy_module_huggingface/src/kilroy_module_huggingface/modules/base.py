from abc import ABC
from typing import Generic, TypeVar

from kilroy_module_server_py_sdk import (
    Categorizable,
    Module,
    classproperty,
    normalize,
)

StateType = TypeVar("StateType")


class HuggingfaceModule(
    Categorizable, Module[StateType], Generic[StateType], ABC
):
    @classproperty
    def category(cls) -> str:
        name: str = cls.__name__
        return normalize(name.removesuffix("HuggingfaceModule"))
