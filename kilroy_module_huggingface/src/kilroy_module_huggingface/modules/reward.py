import asyncio
import json
import logging
from asyncio import Queue
from collections import Counter
from functools import partial
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
)
from uuid import UUID

from kilroy_module_pytorch_py_sdk import (
    Codec,
    Generator,
    Metadata,
    Optimizer,
    RewardModelModule,
    RewardModelModuleLanguageModelState as LanguageModelState,
    RewardModelModuleMetricsState as MetricsState,
    RewardModelModuleReportsState as ReportsState,
    RewardModelModuleRewardModelState as RewardModelState,
    RewardModelModuleState as State,
    Savable,
    Scheduler,
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

logger = logging.getLogger(__name__)


class ModelParams(SerializableModel):
    name: str
    freeze: Optional[str] = None
    optimizer_type: str = "adam"
    optimizers_params: Dict[str, Dict[str, Any]] = {}
    scheduler_type: Optional[str] = None
    schedulers_params: Dict[str, Dict[str, Any]] = {}


class Params(SerializableModel):
    language_model_params: ModelParams
    reward_model_params: ModelParams
    frontend_generator_params: Dict[str, Any] = {}
    backend_generator_params: Dict[str, Any] = {}
    codec_params: Dict[str, Any] = {}
    batch_size: int
    sample_size: int


class RewardModelHuggingfaceModule(
    RewardModelModule, HuggingfaceModule[State]
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
        return await cls._build_categorizable(
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
        return await cls._build_categorizable(
            Optimizer,
            params.reward_model_params.optimizer_type,
            parameters=model.parameters(),
            **params.reward_model_params.optimizers_params.get(
                params.reward_model_params.optimizer_type, {}
            ),
        )

    @classmethod
    async def _build_language_model_scheduler(
        cls, params: Params, optimizer: Optimizer
    ) -> Optional[Scheduler]:
        if params.language_model_params.scheduler_type is None:
            return None
        return await cls._build_categorizable(
            Scheduler,
            params.language_model_params.scheduler_type,
            optimizer=await optimizer.get(),
            **params.language_model_params.schedulers_params.get(
                params.language_model_params.scheduler_type, {}
            ),
        )

    @classmethod
    async def _build_reward_model_scheduler(
        cls, params: Params, optimizer: Optimizer
    ) -> Optional[Scheduler]:
        if params.reward_model_params.scheduler_type is None:
            return None
        return await cls._build_categorizable(
            Scheduler,
            params.reward_model_params.scheduler_type,
            optimizer=await optimizer.get(),
            **params.reward_model_params.schedulers_params.get(
                params.reward_model_params.scheduler_type, {}
            ),
        )

    @classmethod
    async def _build_language_model_state(
        cls, params: Params
    ) -> LanguageModelState:
        model = await cls._build_language_model(params)
        optimizer = await cls._build_language_model_optimizer(params, model)
        model.freeze(params.language_model_params.freeze)
        return LanguageModelState(
            model=model,
            tokenizer=await cls._build_language_model_tokenizer(params),
            optimizer=optimizer,
            optimizers_params=params.language_model_params.optimizers_params,
            scheduler=await cls._build_language_model_scheduler(
                params, optimizer
            ),
            schedulers_params=params.language_model_params.schedulers_params,
        )

    @classmethod
    async def _build_reward_model_state(
        cls, params: Params
    ) -> RewardModelState:
        model = await cls._build_reward_model(params)
        optimizer = await cls._build_reward_model_optimizer(params, model)
        model.freeze(params.reward_model_params.freeze)
        return RewardModelState(
            model=model,
            tokenizer=await cls._build_reward_model_tokenizer(params),
            optimizer=optimizer,
            optimizers_params=params.reward_model_params.optimizers_params,
            scheduler=await cls._build_reward_model_scheduler(
                params, optimizer
            ),
            schedulers_params=params.reward_model_params.schedulers_params,
        )

    @classmethod
    async def _build_frontend_generator(cls, params: Params) -> Generator:
        return await cls._build_configurable(
            Generator, **params.frontend_generator_params
        )

    @classmethod
    async def _build_backend_generator(cls, params: Params) -> Generator:
        return await cls._build_configurable(
            Generator, **params.backend_generator_params
        )

    @classmethod
    async def _build_codec(cls, params: Params) -> Codec:
        return await cls._build_configurable(Codec, **params.codec_params)

    @staticmethod
    async def _build_metrics() -> MetricsState:
        return MetricsState(
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            reward_model_loss_metric=await RewardModelLossMetric.build(),
            reward_model_score_metric=await RewardModelScoreMetric.build(),
        )

    @staticmethod
    async def _build_reports() -> ReportsState:
        return ReportsState(
            step_supervised_losses=[],
            step_reinforced_scores=[],
            step_reward_model_losses=[],
            step_reward_model_scores=[],
        )

    async def _build_default_state(self) -> State:
        params = Params(**self._kwargs)
        coroutine_queue = Queue()
        return State(
            language_model=await self._build_language_model_state(params),
            reward_model=await self._build_reward_model_state(params),
            frontend_generator=await self._build_frontend_generator(params),
            backend_generator=await self._build_backend_generator(params),
            codec=await self._build_codec(params),
            results_cache={},
            batch_size=params.batch_size,
            sample_size=params.sample_size,
            step=0,
            metrics=await self._build_metrics(),
            reports=await self._build_reports(),
            coroutine_queue=coroutine_queue,
            worker_task=asyncio.create_task(self._work(coroutine_queue)),
        )

    @staticmethod
    async def _save_language_model_state(
        state: State, directory: Path
    ) -> None:
        directory = directory / "language_model"
        directory.mkdir(parents=True, exist_ok=True)

        if isinstance(state.language_model, Savable):
            await state.language_model.save(directory / "model")
        if isinstance(state.language_model.tokenizer, Savable):
            await state.language_model.tokenizer.save(directory / "tokenizer")
        if isinstance(state.language_model.optimizer, Savable):
            await state.language_model.optimizer.save(directory / "optimizer")
        if isinstance(state.language_model.scheduler, Savable):
            await state.language_model.scheduler.save(directory / "scheduler")

    @staticmethod
    async def _save_reward_model_state(state: State, directory: Path) -> None:
        directory = directory / "reward_model"
        directory.mkdir(parents=True, exist_ok=True)

        if isinstance(state.language_model, Savable):
            await state.language_model.save(directory / "model")
        if isinstance(state.language_model.tokenizer, Savable):
            await state.language_model.tokenizer.save(directory / "tokenizer")
        if isinstance(state.language_model.optimizer, Savable):
            await state.language_model.optimizer.save(directory / "optimizer")
        if isinstance(state.language_model.scheduler, Savable):
            await state.language_model.scheduler.save(directory / "scheduler")

    @staticmethod
    async def _save_frontend_generator(state: State, directory: Path) -> None:
        await state.frontend_generator.save(directory / "frontend_generator")

    @staticmethod
    async def _save_backend_generator(state: State, directory: Path) -> None:
        await state.backend_generator.save(directory / "backend_generator")

    @staticmethod
    async def _save_codec(state: State, directory: Path) -> None:
        await state.codec.save(directory / "codec")

    @staticmethod
    async def _create_state_dict(state: State) -> Dict[str, Any]:
        return {
            "language_model_optimizer_type": state.language_model.optimizer.category,
            "language_model_optimizers_params": state.language_model.optimizers_params,
            "reward_model_optimizer_type": state.reward_model.optimizer.category,
            "reward_model_optimizers_params": state.reward_model.optimizers_params,
            "language_model_scheduler_type": state.language_model.scheduler.category
            if state.language_model.scheduler is not None
            else None,
            "language_model_schedulers_params": state.language_model.schedulers_params,
            "reward_model_scheduler_type": state.reward_model.scheduler.category
            if state.reward_model.scheduler is not None
            else None,
            "reward_model_schedulers_params": state.reward_model.schedulers_params,
            "batch_size": state.batch_size,
            "sample_size": state.sample_size,
            "step": state.step,
            "step_supervised_losses": state.reports.step_supervised_losses,
            "step_reinforced_scores": state.reports.step_reinforced_scores,
            "step_reward_model_losses": state.reports.step_reward_model_losses,
            "step_reward_model_scores": state.reports.step_reward_model_scores,
        }

    @staticmethod
    async def _save_state_dict(
        directory: Path, state_dict: Dict[str, Any]
    ) -> None:
        with open(directory / "state.json", "w") as f:
            json.dump(state_dict, f)

    @classmethod
    async def _save_state(cls, state: State, directory: Path) -> None:
        await cls._save_language_model_state(state, directory)
        await cls._save_reward_model_state(state, directory)
        await cls._save_frontend_generator(state, directory)
        await cls._save_backend_generator(state, directory)
        await cls._save_codec(state, directory)

        state_dict = await cls._create_state_dict(state)
        await cls._save_state_dict(directory, state_dict)

    @staticmethod
    async def _load_state_dict(directory: Path) -> Dict[str, Any]:
        with open(directory / "state.json", "r") as f:
            return json.load(f)

    @classmethod
    async def _load_language_model(
        cls, directory: Path, params: Params
    ) -> HuggingfaceLanguageModel:
        return await cls._load_generic(
            directory / "language_model" / "model",
            HuggingfaceLanguageModel,
            default=partial(cls._build_language_model, params),
        )

    @classmethod
    async def _load_language_model_optimizer(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        model: HuggingfaceLanguageModel,
    ) -> Optimizer:
        return await cls._load_generic(
            directory / "language_model" / "optimizer",
            Optimizer,
            category=state_dict["language_model_optimizer_type"],
            default=partial(
                cls._build_language_model_optimizer,
                params,
                model,
            ),
            parameters=model.parameters(),
        )

    @classmethod
    async def _load_language_model_tokenizer(
        cls, directory: Path, params: Params
    ) -> HuggingfaceTokenizer:
        return await cls._load_generic(
            directory / "language_model" / "tokenizer",
            HuggingfaceTokenizer,
            default=partial(cls._build_language_model_tokenizer, params),
        )

    @classmethod
    async def _load_language_model_scheduler(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        optimizer: Optimizer,
    ) -> Optional[Scheduler]:
        if state_dict["language_model_scheduler_type"] is None:
            return None
        return await cls._load_generic(
            directory / "language_model" / "scheduler",
            Scheduler,
            category=state_dict["language_model_scheduler_type"],
            default=partial(
                cls._build_language_model_scheduler,
                params,
                optimizer,
            ),
            optimizer=await optimizer.get(),
        )

    @classmethod
    async def _load_language_model_state(
        cls, directory: Path, state_dict: Dict[str, Any], params: Params
    ) -> LanguageModelState:
        model = await cls._load_language_model(directory, params)
        optimizer = await cls._load_language_model_optimizer(
            directory, state_dict, params, model
        )
        return LanguageModelState(
            model=model,
            tokenizer=await cls._load_language_model_tokenizer(
                directory, params
            ),
            optimizer=optimizer,
            optimizers_params=state_dict["language_model_optimizers_params"],
            scheduler=await cls._load_language_model_scheduler(
                directory, state_dict, params, optimizer
            ),
            schedulers_params=state_dict["language_model_schedulers_params"],
        )

    @classmethod
    async def _load_reward_model(
        cls, directory: Path, params: Params
    ) -> HuggingfaceRegressionModel:
        return await cls._load_generic(
            directory / "reward_model" / "model",
            HuggingfaceRegressionModel,
            default=partial(cls._build_reward_model, params),
        )

    @classmethod
    async def _load_reward_model_optimizer(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        model: HuggingfaceRegressionModel,
    ) -> Optimizer:
        return await cls._load_generic(
            directory / "reward_model" / "optimizer",
            Optimizer,
            category=state_dict["reward_model_optimizer_type"],
            default=partial(
                cls._build_reward_model_optimizer,
                params,
                model,
            ),
            parameters=model.parameters(),
        )

    @classmethod
    async def _load_reward_model_tokenizer(
        cls, directory: Path, params: Params
    ) -> HuggingfaceTokenizer:
        return await cls._load_generic(
            directory / "reward_model" / "tokenizer",
            HuggingfaceTokenizer,
            default=partial(cls._build_reward_model_tokenizer, params),
        )

    @classmethod
    async def _load_reward_model_scheduler(
        cls,
        directory: Path,
        state_dict: Dict[str, Any],
        params: Params,
        optimizer: Optimizer,
    ) -> Optional[Scheduler]:
        if state_dict["reward_model_scheduler_type"] is None:
            return None
        return await cls._load_generic(
            directory / "reward_model" / "scheduler",
            Scheduler,
            category=state_dict["reward_model_scheduler_type"],
            default=partial(
                cls._build_reward_model_scheduler,
                params,
                optimizer,
            ),
            optimizer=await optimizer.get(),
        )

    @classmethod
    async def _load_reward_model_state(
        cls, directory: Path, state_dict: Dict[str, Any], params: Params
    ) -> RewardModelState:
        model = await cls._load_reward_model(directory, params)
        optimizer = await cls._load_reward_model_optimizer(
            directory, state_dict, params, model
        )
        return RewardModelState(
            model=model,
            tokenizer=await cls._load_reward_model_tokenizer(
                directory, params
            ),
            optimizer=optimizer,
            optimizers_params=state_dict["reward_model_optimizers_params"],
            scheduler=await cls._load_reward_model_scheduler(
                directory, state_dict, params, optimizer
            ),
            schedulers_params=state_dict["reward_model_schedulers_params"],
        )

    @classmethod
    async def _load_frontend_generator(
        cls, directory: Path, params: Params
    ) -> Generator:
        return await cls._load_generic(
            directory / "frontend_generator",
            Generator,
            default=partial(cls._build_frontend_generator, params),
        )

    @classmethod
    async def _load_backend_generator(
        cls, directory: Path, params: Params
    ) -> Generator:
        return await cls._load_generic(
            directory / "backend_generator",
            Generator,
            default=partial(cls._build_backend_generator, params),
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
        coroutine_queue = Queue()
        return State(
            language_model=await self._load_language_model_state(
                directory, state_dict, params
            ),
            reward_model=await self._load_reward_model_state(
                directory, state_dict, params
            ),
            frontend_generator=await self._load_frontend_generator(
                directory, params
            ),
            backend_generator=await self._load_backend_generator(
                directory, params
            ),
            codec=await self._load_codec(directory, params),
            results_cache={},
            batch_size=state_dict["batch_size"],
            sample_size=state_dict["sample_size"],
            step=state_dict["step"],
            metrics=await self._build_metrics(),
            reports=ReportsState(
                step_supervised_losses=state_dict["step_supervised_losses"],
                step_reinforced_scores=state_dict["step_reinforced_scores"],
                step_reward_model_losses=state_dict[
                    "step_reward_model_losses"
                ],
                step_reward_model_scores=state_dict[
                    "step_reward_model_scores"
                ],
            ),
            coroutine_queue=coroutine_queue,
            worker_task=asyncio.create_task(self._work(coroutine_queue)),
        )

    @staticmethod
    async def _work(queue: Queue[Coroutine]) -> None:
        while True:
            coroutine = await queue.get()
            await coroutine

    async def generate(
        self, n: int, dry: bool
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        logger.info(f"Generating {n} posts...")

        async for post_id, post in super().generate(n, dry):
            logger.info(f"Generated post {str(post_id)}.")
            yield post_id, post

        logger.info("Finished generating posts.")

    async def fit_posts(
        self, posts: AsyncIterable[Tuple[Dict[str, Any], float]]
    ) -> None:
        logger.info("Fitting posts...")

        posts_counter = Counter(posts=0)

        async def count_posts(
            _posts: AsyncIterable[Tuple[Dict[str, Any], float]],
            _posts_counter: Counter,
        ) -> AsyncIterable[Tuple[Dict[str, Any], float]]:
            async for post, score in _posts:
                _posts_counter["posts"] += 1
                yield post, score

        await super().fit_posts(count_posts(posts, posts_counter))
        logger.info(f"Finished fitting {posts_counter['posts']} posts.")

    async def _fit_with_reward_model(self) -> None:
        logger.info("Fitting with reward model...")
        await super()._fit_with_reward_model()
        logger.info("Finished fitting with reward model.")

    async def fit_scores(self, scores: List[Tuple[UUID, float]]) -> None:
        logger.info(f"Fitting {len(scores)} scores...")
        await super().fit_scores(scores)
        logger.info("Finished fitting scores.")

    async def step(self) -> None:
        logger.info("Performing step...")
        await super().step()
        logger.info("Finished performing step.")
