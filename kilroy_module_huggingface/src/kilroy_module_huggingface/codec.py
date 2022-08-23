import json
from typing import Any, Dict, Optional

import torch
from kilroy_module_server_py_sdk import (
    Configurable,
    Parameter,
    SerializableState,
    TextData,
    TextOnlyPost,
    background,
    classproperty,
)
from torch import Tensor

from kilroy_module_huggingface.models import HuggingfaceLanguageModel


class State(SerializableState):
    max_characters: Optional[int] = None


class Codec(Configurable[State]):
    class MaxCharactersParameter(Parameter[State, Optional[int]]):
        @classproperty
        def schema(cls) -> Dict[str, Any]:
            return {
                "type": ["integer", "null"],
                "minimum": 0,
            }

    async def encode(
        self, model: HuggingfaceLanguageModel, sequence: Tensor
    ) -> Dict[str, Any]:
        indices = sequence.flatten().tolist()

        text = await background(
            model.tokenizer.decode,
            indices,
            skip_special_tokens=True,
        )

        async with self.state.read_lock() as state:
            text = text[: state.max_characters]

        post = TextOnlyPost(text=TextData(content=text))
        return json.loads(post.json())

    async def decode(
        self, model: HuggingfaceLanguageModel, post: Dict[str, Any]
    ) -> Tensor:
        post = TextOnlyPost.parse_obj(post)
        text = post.text.content

        async with self.state.read_lock() as state:
            text = text[: state.max_characters]

        indices = await background(model.tokenizer.encode, text)
        if indices or indices[0] != model.tokenizer.bos_token_id:
            indices = [model.tokenizer.bos_token_id] + indices
        if indices or indices[-1] != model.tokenizer.eos_token_id:
            indices = indices + [model.tokenizer.eos_token_id]
        return torch.tensor(indices).view(-1, 1)
