from typing import Any, Dict, Optional

from kilroy_module_pytorch_py_sdk import (
    BasicModule,
    BasicModuleState,
    Codec,
    Configurable,
    Generator,
    Metadata,
    Optimizer,
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

    @staticmethod
    async def _build_optimizer(
        params: Params, model: HuggingfaceLanguageModel
    ) -> Optimizer:
        opt_cls = Optimizer.for_category(params.optimizer_type)
        opt_params = params.optimizers_params.get(params.optimizer_type, {})
        if issubclass(opt_cls, Configurable):
            optimizer = await opt_cls.build(
                params=model.parameters(), **opt_params
            )
            await optimizer.init()
        else:
            optimizer = opt_cls(model.parameters(), **opt_params)
        return optimizer

    @staticmethod
    async def _build_generator(params: Params) -> Generator:
        generator = await Generator.build(**params.generator_params)
        await generator.init()
        return generator

    @staticmethod
    async def _build_codec(params: Params) -> Codec:
        codec = await Codec.build(**params.codec_params)
        await codec.init()
        return codec

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
