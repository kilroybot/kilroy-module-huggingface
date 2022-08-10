import json
from dataclasses import dataclass
from typing import Optional

import torch
from kilroy_module_py_shared import JSON
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    TextData,
    TextOnlyPost,
    classproperty,
)
from kilroy_ws_server_py_sdk import SerializableModel, background
from kilroy_ws_server_py_sdk.configurable import StateType
from tokenizers import Tokenizer
from torch import Tensor


class CodecParams(SerializableModel):
    max_characters: Optional[int] = None


@dataclass
class CodecState:
    tokenizer: Tokenizer
    max_characters: Optional[int]


class Codec(Configurable[CodecState]):
    class MaxCharactersParameter(Parameter[CodecState, Optional[int]]):
        @classproperty
        def schema(cls) -> JSON:
            return {
                "type": ["integer", "null"],
                "minimum": 0,
            }

    def __init__(self, tokenizer: Tokenizer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tokenizer = tokenizer

    async def build_default_state(self) -> StateType:
        params = CodecParams.parse_obj(self._kwargs)
        return CodecState(
            tokenizer=self._tokenizer,
            max_characters=params.max_characters,
        )

    async def encode(self, sequence: Tensor) -> JSON:
        indices = sequence.flatten().tolist()
        async with self.state.read_lock() as state:
            text = await background(
                state.tokenizer.decode,
                indices,
                skip_special_tokens=True,
            )
            text = text[: state.max_characters]
        post = TextOnlyPost(text=TextData(content=text))
        return json.loads(post.json())

    async def decode(self, post: JSON) -> Tensor:
        post = TextOnlyPost.parse_obj(post)
        text = post.text.content
        async with self.state.read_lock() as state:
            text = text[: state.max_characters]
            indices = await background(state.tokenizer.encode, text)
            if indices or indices[0] != state.tokenizer.bos_token_id:
                indices = [state.tokenizer.bos_token_id] + indices
            if indices or indices[-1] != state.tokenizer.eos_token_id:
                indices = indices + [state.tokenizer.eos_token_id]
        return torch.tensor(indices).view(-1, 1)
