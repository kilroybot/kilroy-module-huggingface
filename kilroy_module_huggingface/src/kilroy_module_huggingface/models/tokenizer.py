from pathlib import Path
from typing import List

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kilroy_module_pytorch_py_sdk import Savable, Tokenizer
from kilroy_server_py_utils import background


class HuggingfaceTokenizer(Savable, Tokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    async def save(self, directory: Path) -> None:
        self._tokenizer.save_pretrained(directory)

    @classmethod
    async def from_saved(
        cls, directory: Path, **kwargs
    ) -> "HuggingfaceTokenizer":
        tokenizer = await background(AutoTokenizer.from_pretrained, directory)
        return cls(tokenizer)

    @classmethod
    async def from_path(cls, path: str) -> "HuggingfaceTokenizer":
        tokenizer = await background(AutoTokenizer.from_pretrained, path)
        return cls(tokenizer)

    def encode(self, text: str) -> List[int]:
        indices = self._tokenizer.encode(text)

        if len(indices) <= 1:
            indices = (
                [self._tokenizer.bos_token_id]
                + indices
                + [self._tokenizer.eos_token_id]
            )

        if indices[0] != self._tokenizer.bos_token_id:
            indices = [self._tokenizer.bos_token_id] + indices
        if indices[-1] != self._tokenizer.eos_token_id:
            indices = indices + [self._tokenizer.eos_token_id]

        return indices

    def decode(self, indices: List[int]) -> str:
        return self._tokenizer.decode(indices, skip_special_tokens=True)

    @property
    def start_token(self) -> int:
        return self._tokenizer.bos_token_id

    @property
    def end_token(self) -> int:
        return self._tokenizer.eos_token_id
