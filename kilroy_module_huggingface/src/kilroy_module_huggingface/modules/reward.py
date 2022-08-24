import asyncio
from asyncio import Queue
from typing import Any, Coroutine, Dict, Optional

from kilroy_module_pytorch_py_sdk import (
    Codec,
    Configurable,
    Generator,
    Metadata,
    Optimizer,
    RewardModelModule,
    RewardModelModuleState,
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

    @staticmethod
    async def _build_language_model_optimizer(
        params: Params, model: HuggingfaceLanguageModel
    ) -> Optimizer:
        opt_cls = Optimizer.for_category(
            params.language_model_params.optimizer_type
        )
        opt_params = params.language_model_params.optimizers_params.get(
            params.language_model_params.optimizer_type, {}
        )
        if issubclass(opt_cls, Configurable):
            optimizer = await opt_cls.build(
                params=model.parameters(), **opt_params
            )
            await optimizer.init()
        else:
            optimizer = opt_cls(model.parameters(), **opt_params)
        return optimizer

    @staticmethod
    async def _build_reward_model_optimizer(
        params: Params, model: HuggingfaceRegressionModel
    ) -> Optimizer:
        opt_cls = Optimizer.for_category(
            params.reward_model_params.optimizer_type
        )
        opt_params = params.reward_model_params.optimizers_params.get(
            params.reward_model_params.optimizer_type, {}
        )
        if issubclass(opt_cls, Configurable):
            optimizer = await opt_cls.build(
                params=model.parameters(), **opt_params
            )
            await optimizer.init()
        else:
            optimizer = opt_cls(model.parameters(), **opt_params)
        return optimizer

    @staticmethod
    async def _build_frontend_generator(params: Params) -> Generator:
        generator = await Generator.build(**params.frontend_generator_params)
        await generator.init()
        return generator

    @staticmethod
    async def _build_backend_generator(params: Params) -> Generator:
        generator = await Generator.build(**params.backend_generator_params)
        await generator.init()
        return generator

    @staticmethod
    async def _build_codec(params: Params) -> Codec:
        codec = await Codec.build(**params.codec_params)
        await codec.init()
        return codec

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

    @staticmethod
    async def _work(queue: Queue[Coroutine]) -> None:
        while True:
            coroutine = await queue.get()
            await coroutine
