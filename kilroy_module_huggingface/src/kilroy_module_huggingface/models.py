from typing import Type, TypeVar

from kilroy_module_pytorch_py_sdk import pack_padded, unpack_to_padded
from kilroy_module_server_py_sdk import background
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

T = TypeVar("T")


class HuggingfaceModel(nn.Module):
    def __init__(self, path: str) -> None:
        super().__init__()
        self._model = AutoModelForCausalLM.from_pretrained(path)
        self._tokenizer = AutoTokenizer.from_pretrained(path)

    @classmethod
    async def build(cls: Type[T], *args, **kwargs) -> T:
        return await background(cls, *args, **kwargs)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def _pad_idx(self) -> int:
        if self._tokenizer.pad_token_id is None:
            return 0
        return self.tokenizer.pad_token_id

    def forward(self, x: PackedSequence) -> PackedSequence:
        x, lengths = unpack_to_padded(x, pad_value=self._pad_idx)
        x = x[:, :, 0]
        mask = x != self._pad_idx
        y = self._model(x, attention_mask=mask)
        return pack_padded(y.logits.log_softmax(-1), lengths)
