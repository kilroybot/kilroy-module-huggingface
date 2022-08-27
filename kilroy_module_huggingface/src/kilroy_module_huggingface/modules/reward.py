import asyncio
import json
from asyncio import Queue
from pathlib import Path
from typing import Any, Coroutine, Dict, Optional

from kilroy_module_pytorch_py_sdk import (
    Codec,
    Generator,
    Metadata,
    Optimizer,
    RewardModelModule,
    RewardModelModuleState,
    Savable,
    SerializableModel,
    background,
    classproperty,
)
from kilroy_module_pytorch_py_sdk.modules.reward import (
    ReinforcedScoreMetric,
    RewardModelLossMetric,
    RewardModelScoreMetric,
    SupervisedLossMetric,
)

from kilroy_module_huggingface.models import (
    HuggingfaceLanguageModel,
    HuggingfaceRegressionModel,
)
from kilroy_module_huggingface.modules.base import HuggingfaceModule
from kilroy_module_huggingface.tokenizer import HuggingfaceTokenizer


class ModelParams(SerializableModel):
    name: str
    freeze: Optional[str] = None
    optimizer_type: str = "adam"
    optimizers_params: Dict[str, Dict[str, Any]] = {}


class Params(SerializableModel):
    language_model_params: ModelParams
    reward_model_params: ModelParams
    frontend_generator_params: Dict[str, Any] = {}
    backend_generator_params: Dict[str, Any] = {}
    codec_params: Dict[str, Any] = {}
    batch_size: int
    sample_size: int


class RewardModelHuggingfaceModule(
    RewardModelModule, HuggingfaceModule[RewardModelModuleState]
):
    @classproperty
    def metadata(cls) -> Metadata:
        return Metadata(
            key="kilroy-module-huggingface",
            description="Kilroy module for Huggingface models",
        )

    @staticmethod
    async def _build_language_model(
        params: Params,
    ) -> HuggingfaceLanguageModel:
        return await background(
            HuggingfaceLanguageModel.from_path,
            params.language_model_params.name,
        )

    @staticmethod
    async def _build_reward_model(
        params: Params,
    ) -> HuggingfaceRegressionModel:
        return await background(
            HuggingfaceRegressionModel.from_path,
            params.reward_model_params.name,
        )

    @staticmethod
    async def _build_language_model_tokenizer(
        params: Params,
    ) -> HuggingfaceTokenizer:
        return await background(
            HuggingfaceTokenizer.from_path, params.language_model_params.name
        )

    @staticmethod
    async def _build_reward_model_tokenizer(
        params: Params,
    ) -> HuggingfaceTokenizer:
        return await background(
            HuggingfaceTokenizer.from_path, params.reward_model_params.name
        )

    @classmethod
    async def _build_language_model_optimizer(
        cls, params: Params, model: HuggingfaceLanguageModel
    ) -> Optimizer:
        return await cls.build_categorizable(
            Optimizer,
            params.language_model_params.optimizer_type,
            parameters=model.parameters(),
            **params.language_model_params.optimizers_params.get(
                params.language_model_params.optimizer_type, {}
            ),
        )

    @classmethod
    async def _build_reward_model_optimizer(
        cls, params: Params, model: HuggingfaceRegressionModel
    ) -> Optimizer:
        return await cls.build_categorizable(
            Optimizer,
            params.reward_model_params.optimizer_type,
            parameters=model.parameters(),
            **params.reward_model_params.optimizers_params.get(
                params.reward_model_params.optimizer_type, {}
            ),
        )

    @classmethod
    async def _build_frontend_generator(cls, params: Params) -> Generator:
        return await cls.build_configurable(
            Generator, **params.frontend_generator_params
        )

    @classmethod
    async def _build_backend_generator(cls, params: Params) -> Generator:
        return await cls.build_configurable(
            Generator, **params.backend_generator_params
        )

    @classmethod
    async def _build_codec(cls, params: Params) -> Codec:
        return await cls.build_configurable(Codec, **params.codec_params)

    async def build_default_state(self) -> RewardModelModuleState:
        params = Params(**self._kwargs)
        language_model = await self._build_language_model(params)
        reward_model = await self._build_reward_model(params)
        language_model_optimizer = await self._build_language_model_optimizer(
            params, language_model
        )
        reward_model_optimizer = await self._build_reward_model_optimizer(
            params, reward_model
        )
        language_model.freeze(params.language_model_params.freeze)
        reward_model.freeze(params.reward_model_params.freeze)
        coroutine_queue = Queue()
        return RewardModelModuleState(
            language_model=language_model,
            reward_model=reward_model,
            language_model_tokenizer=await self._build_language_model_tokenizer(
                params
            ),
            reward_model_tokenizer=await self._build_reward_model_tokenizer(
                params
            ),
            language_model_optimizer=language_model_optimizer,
            language_model_optimizers_params=params.language_model_params.optimizers_params,
            reward_model_optimizer=reward_model_optimizer,
            reward_model_optimizers_params=params.reward_model_params.optimizers_params,
            frontend_generator=await self._build_frontend_generator(params),
            backend_generator=await self._build_backend_generator(params),
            codec=await self._build_codec(params),
            results_cache={},
            batch_size=params.batch_size,
            sample_size=params.sample_size,
            epoch=0,
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            reward_model_loss_metric=await RewardModelLossMetric.build(),
            reward_model_score_metric=await RewardModelScoreMetric.build(),
            epoch_supervised_losses=[],
            epoch_reinforced_scores=[],
            epoch_reward_model_losses=[],
            epoch_reward_model_scores=[],
            coroutine_queue=coroutine_queue,
            worker_task=asyncio.create_task(self._work(coroutine_queue)),
        )

    @classmethod
    async def save_state(
        cls, state: RewardModelModuleState, directory: Path
    ) -> None:
        lm_dir = directory / "language_model"
        lm_dir.mkdir(parents=True, exist_ok=True)
        rm_dir = directory / "reward_model"
        rm_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(state.language_model, Savable):
            await state.language_model.save(lm_dir / "model")
        if isinstance(state.language_model_tokenizer, Savable):
            await state.language_model_tokenizer.save(lm_dir / "tokenizer")
        if isinstance(state.language_model_optimizer, Savable):
            await state.language_model_optimizer.save(lm_dir / "optimizer")

        if isinstance(state.reward_model, Savable):
            await state.reward_model.save(rm_dir / "model")
        if isinstance(state.reward_model_tokenizer, Savable):
            await state.reward_model_tokenizer.save(rm_dir / "tokenizer")
        if isinstance(state.reward_model_optimizer, Savable):
            await state.reward_model_optimizer.save(rm_dir / "optimizer")

        await state.frontend_generator.save(directory / "frontend_generator")
        await state.backend_generator.save(directory / "backend_generator")
        await state.codec.save(directory / "codec")

        state_dict = {
            "language_model_optimizer_type": state.language_model_optimizer.category,
            "language_model_optimizers_params": state.language_model_optimizers_params,
            "reward_model_optimizer_type": state.reward_model_optimizer.category,
            "reward_model_optimizers_params": state.reward_model_optimizers_params,
            "batch_size": state.batch_size,
            "sample_size": state.sample_size,
            "epoch": state.epoch,
            "epoch_supervised_losses": state.epoch_supervised_losses,
            "epoch_reinforced_scores": state.epoch_reinforced_scores,
            "epoch_reward_model_losses": state.epoch_reward_model_losses,
            "epoch_reward_model_scores": state.epoch_reward_model_scores,
        }

        with (directory / "state.json").open("w") as f:
            json.dump(state_dict, f)

    async def load_saved_state(
        self, directory: Path
    ) -> RewardModelModuleState:
        with (directory / "state.json").open("r") as f:
            state_dict = json.load(f)

        language_model = await self.load_generic(
            directory / "language_model" / "model",
            HuggingfaceLanguageModel,
        )
        reward_model = await self.load_generic(
            directory / "reward_model" / "model",
            HuggingfaceRegressionModel,
        )
        coroutine_queue = Queue()

        return RewardModelModuleState(
            language_model=language_model,
            reward_model=reward_model,
            language_model_tokenizer=await self.load_generic(
                directory / "language_model" / "tokenizer",
                HuggingfaceTokenizer,
            ),
            reward_model_tokenizer=await self.load_generic(
                directory / "reward_model" / "tokenizer",
                HuggingfaceTokenizer,
            ),
            language_model_optimizer=await self.load_generic(
                directory / "language_model" / "optimizer",
                Optimizer,
                category=state_dict["language_model_optimizer_type"],
                parameters=language_model.parameters(),
                **state_dict["language_model_optimizers_params"].get(
                    state_dict["language_model_optimizer_type"], {}
                ),
            ),
            language_model_optimizers_params=state_dict[
                "language_model_optimizers_params"
            ],
            reward_model_optimizer=await self.load_generic(
                directory / "reward_model" / "optimizer",
                Optimizer,
                category=state_dict["reward_model_optimizer_type"],
                parameters=reward_model.parameters(),
                **state_dict["reward_model_optimizers_params"].get(
                    state_dict["reward_model_optimizer_type"], {}
                ),
            ),
            reward_model_optimizers_params=state_dict[
                "reward_model_optimizers_params"
            ],
            frontend_generator=await self.load_generic(
                directory / "frontend_generator", Generator
            ),
            backend_generator=await self.load_generic(
                directory / "backend_generator", Generator
            ),
            codec=await self.load_generic(directory / "codec", Codec),
            results_cache={},
            batch_size=state_dict["batch_size"],
            sample_size=state_dict["sample_size"],
            epoch=state_dict["epoch"],
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            reward_model_loss_metric=await RewardModelLossMetric.build(),
            reward_model_score_metric=await RewardModelScoreMetric.build(),
            epoch_supervised_losses=state_dict["epoch_supervised_losses"],
            epoch_reinforced_scores=state_dict["epoch_reinforced_scores"],
            epoch_reward_model_losses=state_dict["epoch_reward_model_losses"],
            epoch_reward_model_scores=state_dict["epoch_reward_model_scores"],
            coroutine_queue=coroutine_queue,
            worker_task=asyncio.create_task(self._work(coroutine_queue)),
        )

    @staticmethod
    async def _work(queue: Queue[Coroutine]) -> None:
        while True:
            coroutine = await queue.get()
            await coroutine
