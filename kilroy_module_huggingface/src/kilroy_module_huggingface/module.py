from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Set, Tuple
from uuid import UUID, uuid4

import torch
from aiostream import stream
from kilroy_module_pytorch_py_sdk import (
    pack_list,
    unpack_to_list,
)
from kilroy_module_pytorch_py_sdk.utils import (
    truncate_first_element,
    truncate_last_element,
)
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
from torch.nn import NLLLoss

from kilroy_module_huggingface.codec import Codec
from kilroy_module_huggingface.generator import Generator
from kilroy_module_huggingface.models import HuggingfaceModel
from kilroy_module_huggingface.optimizers import Optimizer


class HuggingfaceModuleParams(SerializableModel):
    model_name: str
    optimizer_type: str
    optimizers_params: Dict[str, Dict[str, Any]]
    generator_params: Dict[str, Any]
    codec_params: Dict[str, Any]
    batch_size: int


@dataclass
class HuggingfaceModuleState:
    model: HuggingfaceModel
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    generator: Generator
    codec: Codec
    logprobs_cache: Dict[UUID, Tensor]
    batch_size: int


class OptimizerParameter(
    CategorizableBasedParameter[HuggingfaceModuleState, Optimizer]
):
    async def _get_params(
        self, state: HuggingfaceModuleState, category: str
    ) -> Dict[str, Any]:
        return {
            "params": state.model.parameters(),
            **state.optimizers_params.get(category, {}),
        }


class GeneratorParameter(NestedParameter[HuggingfaceModuleState, Generator]):
    pass


class CodecParameter(NestedParameter[HuggingfaceModuleState, Codec]):
    pass


class HuggingfaceModule(Module[HuggingfaceModuleState]):
    @classproperty
    def metadata(cls) -> Metadata:
        return Metadata(
            key="kilroy-module-huggingface",
            description="Kilroy module using Huggingface models.",
        )

    @classproperty
    def post_schema(cls) -> JSONSchema:
        return JSONSchema(**TextOnlyPost.schema())

    @classproperty
    def metrics(cls) -> Set[Metric]:
        return set()

    @staticmethod
    async def _build_model(
        params: HuggingfaceModuleParams,
    ) -> HuggingfaceModel:
        return await HuggingfaceModel.build(params.model_name)

    @staticmethod
    async def _build_optimizer(
        params: HuggingfaceModuleParams, model: HuggingfaceModel
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
    async def _build_generator(params: HuggingfaceModuleParams) -> Generator:
        generator = await Generator.build(**params.generator_params)
        await generator.init()
        return generator

    @staticmethod
    async def _build_codec(
        params: HuggingfaceModuleParams, model: HuggingfaceModel
    ) -> Codec:
        codec = await Codec.build(
            tokenizer=model.tokenizer, **params.codec_params
        )
        await codec.init()
        return codec

    async def build_default_state(self) -> HuggingfaceModuleState:
        params = HuggingfaceModuleParams(**self._kwargs)
        model = await self._build_model(params)
        return HuggingfaceModuleState(
            model=model,
            optimizer=await self._build_optimizer(params, model),
            optimizers_params=params.optimizers_params,
            generator=await self._build_generator(params),
            codec=await self._build_codec(params, model),
            logprobs_cache={},
            batch_size=params.batch_size,
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            await background(state.buffer.store.__exit__)

    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            OptimizerParameter(),
            GeneratorParameter(),
            CodecParameter(),
        }

    async def generate(
        self, n: int
    ) -> AsyncIterable[Tuple[UUID, Dict[str, Any]]]:
        state = await self.state.value.fetch()
        async for result in state.generator.generate(state.model, n):
            sequences = unpack_to_list(result.sequences)
            for sequence, logprob in zip(sequences, result.logprobs):
                post_id = uuid4()
                post = await state.codec.encode(sequence)
                state.logprobs_cache[post_id] = logprob[0]
                yield post_id, post

    async def fit_posts(self, posts: AsyncIterable[Dict[str, Any]]) -> None:
        async with self.state.read_lock() as state:
            batches = stream.chunks(posts, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                async with self.state.read_lock() as state:
                    sequences = [
                        await state.codec.decode(post) for post in batch
                    ]

                    def fit(model, seq):
                        input = pack_list(truncate_last_element(seq))
                        target = pack_list(truncate_first_element(seq))
                        logprobs = model(input)
                        loss = NLLLoss()(logprobs.data, target.data.flatten())
                        loss.backward()

                    await background(fit, state.model, sequences)

    async def fit_scores(self, scores: List[Tuple[UUID, float]]) -> None:
        # noinspection PyShadowingNames
        def _fit(logprobs, scores):
            loss = -(logprobs * scores).mean()
            loss.backward()

        async with self.state.read_lock() as state:
            cache = state.logprobs_cache

        logprobs = torch.stack([cache.pop(pid) for pid, _ in scores])
        scores = torch.tensor([score for _, score in scores])

        await background(_fit, logprobs, scores)

    async def step(self) -> None:
        def _step(opt):
            opt.step()
            opt.zero_grad()

        async with self.state.read_lock() as state:
            optimizer = await state.optimizer.get()
            await background(_step, optimizer)
