import json
from pathlib import Path
from typing import Any, Dict, Optional

from kilroy_module_pytorch_py_sdk import (
    BasicModule,
    BasicModuleState,
    Codec,
    Generator,
    Metadata,
    Optimizer,
    Savable,
    SerializableModel,
    background,
    classproperty,
)
from kilroy_module_pytorch_py_sdk.modules.basic import (
    ReinforcedScoreMetric,
    SupervisedLossMetric,
)

from kilroy_module_huggingface.models import HuggingfaceLanguageModel
from kilroy_module_huggingface.modules.base import HuggingfaceModule
from kilroy_module_huggingface.tokenizer import HuggingfaceTokenizer


class Params(SerializableModel):
    model_name: str
    freeze: Optional[str] = None
    optimizer_type: str = "adam"
    optimizers_params: Dict[str, Dict[str, Any]] = {}
    generator_params: Dict[str, Any] = {}
    codec_params: Dict[str, Any] = {}
    batch_size: int


class BasicHuggingfaceModule(BasicModule, HuggingfaceModule[BasicModuleState]):
    @classproperty
    def metadata(cls) -> Metadata:
        return Metadata(
            key="kilroy-module-huggingface",
            description="Kilroy module for Huggingface models",
        )

    @staticmethod
    async def _build_model(
        params: Params,
    ) -> HuggingfaceLanguageModel:
        return await background(
            HuggingfaceLanguageModel.from_path, params.model_name
        )

    @staticmethod
    async def _build_tokenizer(
        params: Params,
    ) -> HuggingfaceTokenizer:
        return await background(
            HuggingfaceTokenizer.from_path, params.model_name
        )

    @classmethod
    async def _build_optimizer(
        cls, params: Params, model: HuggingfaceLanguageModel
    ) -> Optimizer:
        return await cls.build_categorizable(
            Optimizer,
            params.optimizer_type,
            parameters=model.parameters(),
            **params.optimizers_params.get(params.optimizer_type, {}),
        )

    @classmethod
    async def _build_generator(cls, params: Params) -> Generator:
        return await cls.build_configurable(
            Generator, **params.generator_params
        )

    @classmethod
    async def _build_codec(cls, params: Params) -> Codec:
        return await cls.build_configurable(Codec, **params.generator_params)

    async def build_default_state(self) -> BasicModuleState:
        params = Params(**self._kwargs)
        model = await self._build_model(params)
        optimizer = await self._build_optimizer(params, model)
        model.freeze(params.freeze)
        return BasicModuleState(
            model=model,
            tokenizer=await self._build_tokenizer(params),
            optimizer=optimizer,
            optimizers_params=params.optimizers_params,
            generator=await self._build_generator(params),
            codec=await self._build_codec(params),
            results_cache={},
            batch_size=params.batch_size,
            epoch=0,
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            epoch_supervised_losses=[],
            epoch_reinforced_scores=[],
        )

    @classmethod
    async def save_state(
        cls, state: BasicModuleState, directory: Path
    ) -> None:
        if isinstance(state.model, Savable):
            await state.model.save(directory / "model")
        if isinstance(state.tokenizer, Savable):
            await state.tokenizer.save(directory / "tokenizer")
        if isinstance(state.optimizer, Savable):
            await state.optimizer.save(directory / "optimizer")
        await state.generator.save(directory / "generator")
        await state.codec.save(directory / "codec")

        state_dict = {
            "optimizer_type": state.optimizer.category,
            "optimizers_params": state.optimizers_params,
            "batch_size": state.batch_size,
            "epoch": state.epoch,
            "epoch_supervised_losses": state.epoch_supervised_losses,
            "epoch_reinforced_scores": state.epoch_reinforced_scores,
        }

        with (directory / "state.json").open("w") as f:
            json.dump(state_dict, f)

    async def load_saved_state(self, directory: Path) -> BasicModuleState:
        with (directory / "state.json").open("r") as f:
            state_dict = json.load(f)

        model = await self.load_generic(
            directory / "model", HuggingfaceLanguageModel
        )

        return BasicModuleState(
            model=model,
            tokenizer=await self.load_generic(
                directory / "tokenizer", HuggingfaceTokenizer
            ),
            optimizer=await self.load_generic(
                directory / "optimizer",
                Optimizer,
                category=state_dict["optimizer_type"],
                parameters=model.parameters(),
                **state_dict["optimizers_params"].get(
                    state_dict["optimizer_type"], {}
                ),
            ),
            optimizers_params=state_dict["optimizers_params"],
            generator=await self.load_generic(
                directory / "generator", Generator
            ),
            codec=await self.load_generic(directory / "codec", Codec),
            results_cache={},
            batch_size=state_dict["batch_size"],
            epoch=state_dict["epoch"],
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            epoch_supervised_losses=state_dict["epoch_supervised_losses"],
            epoch_reinforced_scores=state_dict["epoch_reinforced_scores"],
        )
