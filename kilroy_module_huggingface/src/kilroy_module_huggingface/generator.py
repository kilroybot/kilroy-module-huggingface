import random
from dataclasses import dataclass
from typing import AsyncIterable, Dict, List, Set

from kilroy_module_py_shared import JSON
from kilroy_module_pytorch_py_sdk import (
    GenerationResult,
    SequenceGenerator,
)
from kilroy_module_server_py_sdk import (
    CategorizableBasedParameter,
    Configurable,
    Parameter,
    SerializableModel,
    classproperty,
)
from transformers import PreTrainedTokenizerBase

from kilroy_module_huggingface.models import HuggingfaceModel
from kilroy_module_huggingface.samplers import Sampler


class GeneratorParams(SerializableModel):
    sampler_type: str
    samplers_params: Dict[str, JSON]
    contexts: List[str]
    max_length: int
    end_tokens: List[str]
    batch_size: int


@dataclass
class GeneratorState:
    generator: SequenceGenerator
    sampler: Sampler
    samplers_params: Dict[str, JSON]
    contexts: List[str]
    max_length: int
    end_tokens: List[str]
    batch_size: int


class SamplerParameter(CategorizableBasedParameter[GeneratorState, Sampler]):
    pass


class ContextsParameter(Parameter[GeneratorState, List[str]]):
    @classproperty
    def schema(cls) -> JSON:
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }


class MaxLengthParameter(Parameter[GeneratorState, int]):
    @classproperty
    def schema(cls) -> JSON:
        return {
            "type": "integer",
            "minimum": 1,
        }


class EndTokensParameter(Parameter[GeneratorState, List[str]]):
    @classproperty
    def schema(cls) -> JSON:
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }


class BatchSizeParameter(Parameter[GeneratorState, int]):
    @classproperty
    def schema(cls) -> JSON:
        return {
            "type": "integer",
            "minimum": 1,
        }


class Generator(Configurable[GeneratorState]):
    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            SamplerParameter(),
            ContextsParameter(),
            MaxLengthParameter(),
            EndTokensParameter(),
            BatchSizeParameter(),
        }

    async def build_default_state(self) -> GeneratorState:
        params = GeneratorParams(**self._kwargs)
        sampler_cls = Sampler.for_category(params.sampler_type)
        sampler_params = params.samplers_params.get(params.sampler_type, {})
        if issubclass(sampler_cls, Configurable):
            sampler = await sampler_cls.build(**sampler_params)
            await sampler.init()
        else:
            sampler = sampler_cls(**sampler_params)
        return GeneratorState(
            generator=SequenceGenerator(),
            sampler=sampler,
            samplers_params=params.samplers_params,
            contexts=params.contexts,
            max_length=params.max_length,
            end_tokens=params.end_tokens,
            batch_size=params.batch_size,
        )

    @staticmethod
    def _get_contexts(
        state: GeneratorState, tokenizer: PreTrainedTokenizerBase, n: int
    ) -> List[List[int]]:
        contexts = random.choices(state.contexts, k=n)
        contexts = [
            context
            if context.startswith(tokenizer.bos_token)
            else tokenizer.bos_token + context
            for context in contexts
        ]
        return [tokenizer.encode(context) for context in contexts]

    @staticmethod
    def _get_end_tokens(
        state: GeneratorState, tokenizer: PreTrainedTokenizerBase
    ) -> List[int]:
        tokens = [token or tokenizer.eos_token for token in state.end_tokens]
        return [tokenizer.encode(token)[0] for token in tokens]

    async def generate(
        self,
        model: HuggingfaceModel,
        n: int,
    ) -> AsyncIterable[GenerationResult]:
        async with self.state.read_lock() as state:
            end_tokens = self._get_end_tokens(state, model.tokenizer)
            sampler = await state.sampler.get()

            while n > 0:
                batch_size = min(n, state.batch_size)
                n -= batch_size
                contexts = self._get_contexts(
                    state, model.tokenizer, batch_size
                )

                yield state.generator.generate(
                    model,
                    sampler,
                    contexts,
                    state.max_length,
                    end_tokens,
                )
                # yield await background(
                #     state.generator.generate,
                #     model,
                #     sampler,
                #     contexts,
                #     state.max_length,
                #     end_tokens,
                # )
