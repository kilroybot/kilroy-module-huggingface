from asyncio import Lock
from pathlib import Path
from typing import Type, Optional

from huggingface_hub.utils import HFValidationError

from kilroy_module_huggingface.models.models import (
    HuggingfaceSequentialModelBase,
)
from kilroy_module_huggingface.models.tokenizer import HuggingfaceTokenizer
from kilroy_module_pytorch_py_sdk import ModelLoader, SequentialModel
from kilroy_module_pytorch_py_sdk.models.loader import ModelInfo, ModelType
from kilroy_server_py_utils import Savable


class HuggingfaceModelLoader(ModelLoader[SequentialModel]):
    def __init__(
        self,
        lock: Lock,
        state_directory: Path,
        model_class: Type[HuggingfaceSequentialModelBase],
        path: str,
        batch_size: int,
        freeze: Optional[str] = None,
    ) -> None:
        self._state_directory = state_directory
        self._model_class = model_class
        self._path = path
        self._batch_size = batch_size
        self._freeze = freeze
        super().__init__(lock)

    @property
    def _model_directory(self) -> Path:
        return self._state_directory / "model"

    @property
    def _tokenizer_directory(self) -> Path:
        return self._state_directory / "tokenizer"

    async def _load(self) -> ModelInfo[SequentialModel]:
        try:
            model = await self._model_class.from_saved(self._model_directory)
        except (FileNotFoundError, HFValidationError):
            model = await self._model_class.from_path(self._path)

        try:
            tokenizer = await HuggingfaceTokenizer.from_saved(
                self._tokenizer_directory
            )
        except (FileNotFoundError, HFValidationError):
            tokenizer = await HuggingfaceTokenizer.from_path(self._path)

        if self._freeze is not None:
            model.freeze(self._freeze)

        return ModelInfo(model, tokenizer, self._batch_size, Lock())

    async def _save(self, info: ModelInfo[SequentialModel]) -> None:
        if isinstance(info.model, Savable):
            await info.model.save(self._model_directory)
        if isinstance(info.tokenizer, Savable):
            await info.tokenizer.save(self._tokenizer_directory)

    async def _reset(self) -> ModelInfo[ModelType]:
        model = await self._model_class.from_path(self._path)
        tokenizer = await HuggingfaceTokenizer.from_path(self._path)

        if self._freeze is not None:
            model.freeze(self._freeze)

        return ModelInfo(model, tokenizer, self._batch_size, Lock())
