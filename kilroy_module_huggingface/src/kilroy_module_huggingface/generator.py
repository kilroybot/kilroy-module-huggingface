import random
from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Set

from kilroy_module_pytorch_py_sdk import GenerationResult, SequenceGenerator
from kilroy_module_server_py_sdk import (
    CategorizableBasedParameter,
    Configurable,
    Parameter,
    SerializableModel,
    background,
    classproperty,
)
from transformers import PreTrainedTokenizerBase

from kilroy_module_huggingface.models import HuggingfaceLanguageModel
from kilroy_module_huggingface.samplers import Sampler


class Params(SerializableModel):
    sampler_type: str = "epsilonNucleus"
    samplers_params: Dict[str, Dict[str, Any]] = {}
    contexts: List[str] = [""]
    max_length: int
    end_tokens: List[str] = []
    batch_size: int


@dataclass
class State:
    generator: SequenceGenerator
    sampler: Sampler
    samplers_params: Dict[str, Dict[str, Any]]
    contexts: List[str]
    max_length: int
    end_tokens: List[str]
    batch_size: int


class SamplerParameter(CategorizableBasedParameter[State, Sampler]):
    pass


class ContextsParameter(Parameter[State, List[str]]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }


class MaxLengthParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 1,
        }


class EndTokensParameter(Parameter[State, List[str]]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        }


class BatchSizeParameter(Parameter[State, int]):
    @classproperty
    def schema(cls) -> Dict[str, Any]:
        return {
            "type": "integer",
            "minimum": 1,
        }


class Generator(Configurable[State]):
    @classproperty
    def parameters(cls) -> Set[Parameter]:
        return {
            SamplerParameter(),
            ContextsParameter(),
            MaxLengthParameter(),
            EndTokensParameter(),
            BatchSizeParameter(),
        }

    async def build_default_state(self) -> State:
        params = Params(**self._kwargs)
        sampler_cls = Sampler.for_category(params.sampler_type)
        sampler_params = params.samplers_params.get(params.sampler_type, {})
        if issubclass(sampler_cls, Configurable):
            sampler = await sampler_cls.build(**sampler_params)
            await sampler.init()
        else:
            sampler = sampler_cls(**sampler_params)
        return State(
            generator=SequenceGenerator(),
            sampler=sampler,
            samplers_params=params.samplers_params,
            contexts=params.contexts,
            max_length=params.max_length,
            end_tokens=params.end_tokens,
            batch_size=params.batch_size,
        )

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            if isinstance(state.sampler, Configurable):
                await state.sampler.cleanup()

    @staticmethod
    def _get_contexts(
        state: State, tokenizer: PreTrainedTokenizerBase, n: int
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
        state: State, tokenizer: PreTrainedTokenizerBase
    ) -> List[int]:
        tokens = [token or tokenizer.eos_token for token in state.end_tokens]
        return [tokenizer.encode(token)[0] for token in tokens]

    async def generate(
        self,
        model: HuggingfaceLanguageModel,
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

                yield await background(
                    state.generator.generate,
                    model,
                    sampler,
                    contexts,
                    state.max_length,
                    end_tokens,
                )
