import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from kilroy_module_pytorch_py_sdk import (
    BasicModule,
    BasicModuleMetricsState as MetricsState,
    BasicModuleReportsState as ReportsState,
    BasicModuleState as State,
    Codec,
    Generator,
    Metadata,
    Optimizer,
    Savable,
    Scheduler,
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
    scheduler_type: Optional[str] = None
    schedulers_params: Dict[str, Dict[str, Any]] = {}
    generator_params: Dict[str, Any] = {}
    codec_params: Dict[str, Any] = {}
    batch_size: int


class BasicHuggingfaceModule(BasicModule, HuggingfaceModule[State]):
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
        return await cls._build_categorizable(
            Optimizer,
            params.optimizer_type,
            parameters=model.parameters(),
            **params.optimizers_params.get(params.optimizer_type, {}),
        )

    @classmethod
    async def _build_scheduler(
        cls, params: Params, optimizer: Optimizer
    ) -> Optional[Scheduler]:
        if params.scheduler_type is None:
            return None
        return await cls._build_categorizable(
            Scheduler,
            params.scheduler_type,
            optimizer=await optimizer.get(),
            **params.schedulers_params.get(params.scheduler_type, {}),
        )

    @classmethod
    async def _build_generator(cls, params: Params) -> Generator:
        return await cls._build_configurable(
            Generator, **params.generator_params
        )

    @classmethod
    async def _build_codec(cls, params: Params) -> Codec:
        return await cls._build_configurable(Codec, **params.generator_params)

    @staticmethod
    async def _build_metrics() -> MetricsState:
        return MetricsState(
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
        )

    @staticmethod
    async def _build_reports() -> ReportsState:
        return ReportsState(
            step_reinforced_scores=[], step_supervised_losses=[]
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        model = await self._build_model(params)
        optimizer = await self._build_optimizer(params, model)
        model.freeze(params.freeze)
        return State(
            model=model,
            tokenizer=await self._build_tokenizer(params),
            optimizer=optimizer,
            optimizers_params=params.optimizers_params,
            scheduler=await self._build_scheduler(params, optimizer),
            schedulers_params=params.schedulers_params,
            generator=await self._build_generator(params),
            codec=await self._build_codec(params),
            results_cache={},
            batch_size=params.batch_size,
            step=0,
            metrics=await self._build_metrics(),
            reports=await self._build_reports(),
        )

    @staticmethod
    async def _save_model(state: State, directory: Path) -> None:
        if isinstance(state.model, Savable):
            await state.model.save(directory / "model")

    @staticmethod
    async def _save_tokenizer(state: State, directory: Path) -> None:
        if isinstance(state.tokenizer, Savable):
            await state.tokenizer.save(directory / "tokenizer")

    @staticmethod
    async def _save_optimizer(state: State, directory: Path) -> None:
        if isinstance(state.optimizer, Savable):
            await state.optimizer.save(directory / "optimizer")

    @staticmethod
    async def _save_scheduler(state: State, directory: Path) -> None:
        if isinstance(state.scheduler, Savable):
            await state.scheduler.save(directory / "scheduler")

    @staticmethod
    async def _save_generator(state: State, directory: Path) -> None:
        await state.generator.save(directory / "generator")

    @staticmethod
    async def _save_codec(state: State, directory: Path) -> None:
        await state.codec.save(directory / "codec")

    @staticmethod
    async def _create_state_dict(state: State) -> Dict[str, Any]:
        return {
            "optimizer_type": state.optimizer.category,
            "optimizers_params": state.optimizers_params,
            "scheduler_type": state.scheduler.category
            if state.scheduler is not None
            else None,
            "schedulers_params": state.schedulers_params,
            "batch_size": state.batch_size,
            "step": state.step,
            "step_supervised_losses": state.reports.step_supervised_losses,
            "step_reinforced_scores": state.reports.step_reinforced_scores,
        }

    @staticmethod
    async def _save_state_dict(
        directory: Path, state_dict: Dict[str, Any]
    ) -> None:
        with open(directory / "state.json", "w") as f:
            json.dump(state_dict, f)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_model(state, directory)
        await cls._save_tokenizer(state, directory)
        await cls._save_optimizer(state, directory)
        await cls._save_scheduler(state, directory)
        await cls._save_generator(state, directory)
        await cls._save_codec(state, directory)

        state_dict = await cls._create_state_dict(state)
        await cls._save_state_dict(directory, state_dict)

    @staticmethod
    async def _load_state_dict(directory: Path) -> Dict[str, Any]:
        with open(directory / "state.json", "r") as f:
            return json.load(f)

    @classmethod
    async def _load_model(
        cls, directory: Path, params: Params
    ) -> HuggingfaceLanguageModel:
        return await cls._load_generic(
            directory / "model",
            HuggingfaceLanguageModel,
            default=partial(cls._build_model, params),
        )

    @classmethod
    async def _load_optimizer(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        model: HuggingfaceLanguageModel,
    ) -> Optimizer:
        return await cls._load_generic(
            directory / "optimizer",
            Optimizer,
            category=state_dict["optimizer_type"],
            default=partial(cls._build_optimizer, params, model),
            parameters=model.parameters(),
        )

    @classmethod
    async def _load_tokenizer(
        cls, directory: Path, params: Params
    ) -> HuggingfaceTokenizer:
        return await cls._load_generic(
            directory / "tokenizer",
            HuggingfaceTokenizer,
            default=partial(cls._build_tokenizer, params),
        )

    @classmethod
    async def _load_scheduler(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        optimizer: Optimizer,
    ) -> Optional[Scheduler]:
        if state_dict["scheduler_type"] is None:
            return None
        return await cls._load_generic(
            directory / "scheduler",
            Scheduler,
            category=state_dict["scheduler_type"],
            default=partial(cls._build_scheduler, params, optimizer),
            optimizer=await optimizer.get(),
        )

    @classmethod
    async def _load_generator(
        cls, directory: Path, params: Params
    ) -> Generator:
        return await cls._load_generic(
            directory / "generator",
            Generator,
            default=partial(cls._build_generator, params),
        )

    @classmethod
    async def _load_codec(cls, directory: Path, params: Params) -> Codec:
        return await cls._load_generic(
            directory / "codec",
            Codec,
            default=partial(cls._build_codec, params),
        )

    async def _load_saved_state(self, directory: Path) -> State:
        state_dict = await self._load_state_dict(directory)
        params = Params(**self._kwargs)
        model = await self._load_model(directory, params)
        optimizer = await self._load_optimizer(
            directory, state_dict, params, model
        )

        return State(
            model=model,
            tokenizer=await self._load_tokenizer(directory, params),
            optimizer=optimizer,
            optimizers_params=state_dict["optimizers_params"],
            scheduler=await self._load_scheduler(
                directory, state_dict, params, optimizer
            ),
            schedulers_params=state_dict["schedulers_params"],
            generator=await self._load_generator(directory, params),
            codec=await self._load_codec(directory, params),
            results_cache={},
            batch_size=state_dict["batch_size"],
            step=state_dict["step"],
            metrics=await self._build_metrics(),
            reports=ReportsState(
                step_supervised_losses=state_dict["step_supervised_losses"],
                step_reinforced_scores=state_dict["step_reinforced_scores"],
            ),
        )
