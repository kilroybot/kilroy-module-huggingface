from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

import torch
from kilroy_module_pytorch_py_sdk import unpack_to_list
from kilroy_module_server_py_sdk import (
    CategorizableBasedParameter,
    Configurable,
    JSONSchema,
    Metadata,
    Metric,
    Module,
    NestedParameter,
    Parameter,
    SerializableModel,
    TextOnlyPost,
    background,
    classproperty,
)
from torch import Tensor

from kilroy_module_huggingface.codec import Codec
from kilroy_module_huggingface.generator import Generator
from kilroy_module_huggingface.models import HuggingfaceLanguageModel
from kilroy_module_huggingface.optimizers import Optimizer
from kilroy_module_huggingface.trainers import Trainer


class Params(SerializableModel):
    model_name: str
    freeze: Optional[str] = None
    trainer_type: str = "basic"
    trainers_params: Dict[str, Dict[str, Any]] = {}
    optimizer_type: str = "adam"
    optimizers_params: Dict[str, Dict[str, Any]] = {}
    generator_params: Dict[str, Any] = {}
    codec_params: Dict[str, Any] = {}


@dataclass
class State:
    model: HuggingfaceLanguageModel
    trainer: Trainer
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    generator: Generator
    codec: Codec
    results_cache: Dict[UUID, Tuple[Tensor, Tensor]]


class TrainerParameter(CategorizableBasedParameter[State, Trainer]):
    pass


class OptimizerParameter(CategorizableBasedParameter[State, Optimizer]):
    async def _get_params(self, state: State, category: str) -> Dict[str, Any]:
        return {
            "params": state.model.parameters(),
            **state.optimizers_params.get(category, {}),
        }


class GeneratorParameter(NestedParameter[State, Generator]):
    pass


class CodecParameter(NestedParameter[State, Codec]):
    pass


class HuggingfaceModule(Module[State]):
    @classproperty
    def metadata(cls) -> Metadata:
        return Metadata(
            key="kilroy-module-huggingface",
            description="Kilroy module using Huggingface models.",
        )

    @classproperty
    def post_schema(cls) -> JSONSchema:
        return JSONSchema(**TextOnlyPost.schema())

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return await state.trainer.get_metrics()

    @staticmethod
    async def _build_model(
        params: Params,
    ) -> HuggingfaceLanguageModel:
        return await background(
            HuggingfaceLanguageModel.from_path, params.model_name
        )

    @staticmethod
    async def _build_trainer(params: Params) -> Trainer:
        trainer_cls = Trainer.for_category(params.trainer_type)
        trainer_params = params.trainers_params.get(params.trainer_type, {})
        if issubclass(trainer_cls, Configurable):
            trainer = await trainer_cls.build(**trainer_params)
            await trainer.init()
        else:
            trainer = trainer_cls(**trainer_params)
        return trainer

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

    async def build_default_state(self) -> State:
        params = Params(**self._kwargs)
        model = await self._build_model(params)
        optimizer = await self._build_optimizer(params, model)
        model.freeze(params.freeze)
        return State(
            model=model,
            trainer=await self._build_trainer(params),
            optimizer=optimizer,
            optimizers_params=params.optimizers_params,
            generator=await self._build_generator(params),
            codec=await self._build_codec(params),
            results_cache={},
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.trainer, Configurable):
                await state.trainer.cleanup()
            if isinstance(state.optimizer, Configurable):
                await state.optimizer.cleanup()

    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            TrainerParameter(),
            OptimizerParameter(),
            GeneratorParameter(),
            CodecParameter(),
        }

    async def generate(
        self, n: int
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        async with self.state.read_lock() as state:
            generated = state.generator.generate(state.model, n)

        async for result in generated:
            sequences = unpack_to_list(result.sequences)

            for sequence, logprob in zip(sequences, result.logprobs):
                post_id = uuid4()

                async with self.state.read_lock() as state:
                    post = await state.codec.encode(state.model, sequence)

                async with self.state.write_lock() as state:
                    state.results_cache[post_id] = (sequence, logprob[0])

                yield post_id, post

    async def fit_posts(self, posts: AsyncIterable[Dict[str, Any]]) -> None:
        async with self.state.read_lock() as state:
            trainer = state.trainer
            model = state.model

        async def decoded():
            async for post in posts:
                # noinspection PyShadowingNames
                async with self.state.read_lock() as state:
                    yield await state.codec.decode(state.model, post)

        await trainer.fit_supervised(model, decoded())

    async def fit_scores(self, scores: List[Tuple[UUID, float]]) -> None:
        async with self.state.read_lock() as state:
            trainer = state.trainer
            model = state.model

        async def get_results():
            for post_id, score in scores:
                # noinspection PyShadowingNames
                async with self.state.write_lock() as state:
                    sequence, logprob = state.results_cache.pop(post_id)
                yield sequence, logprob, torch.tensor(score)

        await trainer.fit_reinforced(model, get_results())

    async def step(self) -> None:
        async with self.state.write_lock() as state:
            await state.trainer.step(state.optimizer)
