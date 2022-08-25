from typing import List

from kilroy_module_pytorch_py_sdk import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class HuggingfaceTokenizer(Tokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    @classmethod
    def from_path(cls, path: str) -> "HuggingfaceTokenizer":
        return cls(AutoTokenizer.from_pretrained(path))

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
