import logging
from pathlib import Path
from typing import AsyncIterable, Dict, Any, Tuple, Optional

from kilroy_module_huggingface.models.loader import HuggingfaceModelLoader
from kilroy_module_huggingface.models.models import (
    HuggingfaceLanguageModel,
    HuggingfaceValueModel,
)
from kilroy_module_py_shared import Metadata, SerializableModel
from kilroy_module_pytorch_py_sdk import (
    PytorchModule,
    ModelsRegistry,
    ModelLoader,
    SequentialModel,
)
from kilroy_module_pytorch_py_sdk.module.params import ModelsParams
from kilroy_server_py_utils import classproperty, Configuration
from kilroy_server_py_utils.configurable import StateType


class LoaderParams(SerializableModel):
    path: str
    batch_size: int = 1
    freeze: Optional[str] = None


class HuggingfaceModule(PytorchModule):
    def __init__(
        self, config: Configuration[StateType], state_directory: Path, **kwargs
    ) -> None:
        self._state_directory = state_directory
        super().__init__(config, **kwargs)

    @staticmethod
    async def _build_policy_loader(
        state_directory: Path, params: Dict[str, Any]
    ) -> ModelLoader[SequentialModel]:
        params = LoaderParams(**params)
        return await HuggingfaceModelLoader.create(
            state_directory=state_directory,
            model_class=HuggingfaceLanguageModel,
            path=params.path,
            batch_size=params.batch_size,
            freeze=params.freeze,
        )

    @staticmethod
    async def _build_value_loader(
        state_directory: Path, params: Dict[str, Any]
    ) -> ModelLoader[SequentialModel]:
        params = LoaderParams(**params)
        return await HuggingfaceModelLoader.create(
            state_directory=state_directory,
            model_class=HuggingfaceValueModel,
            path=params.path,
            batch_size=params.batch_size,
            freeze=params.freeze,
        )

    @staticmethod
    async def _build_baseline_loader(
        state_directory: Path, policy_params: Dict[str, Any]
    ) -> ModelLoader[SequentialModel]:
        params = LoaderParams(**policy_params)
        return await HuggingfaceModelLoader.create(
            state_directory=state_directory,
            model_class=HuggingfaceLanguageModel,
            path=params.path,
            batch_size=params.batch_size,
            freeze=".*",
        )

    async def _build_models_registry(
        self, params: ModelsParams
    ) -> ModelsRegistry:
        state_directory = self._state_directory / "models"
        return ModelsRegistry(
            policy=await self._build_policy_loader(
                state_directory / "policy", params.policy
            ),
            value=await self._build_value_loader(
                state_directory / "value", params.value
            ),
            baseline=await self._build_baseline_loader(
                state_directory / "baseline", params.policy
            ),
        )

    # noinspection PyMethodParameters
    @classproperty
    def metadata(cls) -> Metadata:
        return Metadata(
            key="kilroy-module-huggingface",
            description="Kilroy module for Huggingface models",
        )

    # noinspection PyMethodParameters
    @classproperty
    def logger(cls) -> logging.Logger:
        return logging.getLogger(__name__)

    async def generate(
        self, n: int
    ) -> AsyncIterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
        self.logger.info(f"Generating {n} posts...")

        i = 0

        async for content, metadata in super().generate(n):
            self.logger.info(f"Generated post {i}.")
            yield content, metadata
            i += 1

        self.logger.info("Finished generating posts.")

    async def fit_supervised(
        self, data: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        self.logger.info("Fitting supervised...")
        await super().fit_supervised(data)
        self.logger.info(f"Finished fitting supervised.")

    async def fit_reinforced(
        self, data: AsyncIterable[Tuple[Dict[str, Any], Dict[str, Any], float]]
    ) -> None:
        self.logger.info(f"Fitting reinforced...")
        await super().fit_reinforced(data)
        self.logger.info(f"Finished fitting reinforced.")

    async def reset_self(self) -> None:
        self.logger.info("Resetting state...")
        await super().reset_self()
        self.logger.info("State reset.")

    async def save_self(self, directory: Path) -> None:
        self.logger.info("Saving state...")
        await super().save_self(directory)
        self.logger.info("State saved.")
