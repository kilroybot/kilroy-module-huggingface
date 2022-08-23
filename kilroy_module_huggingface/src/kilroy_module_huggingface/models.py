import re
from abc import ABC
from typing import Optional, TypeVar

import torch
from kilroy_module_pytorch_py_sdk import pack_padded, unpack_to_padded
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

T = TypeVar("T")


def make_mask(x: Tensor, lengths: Tensor) -> Tensor:
    indices = torch.arange(x.shape[-1]).repeat((len(x), 1))
    lengths = lengths.view(-1, 1)
    return indices < lengths


class HuggingfaceModelBase(nn.Module, ABC):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._pad_token_id = pad_token_id

    @property
    def base_model(self) -> nn.Module:
        return self._model.base_model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def freeze(self, pattern: Optional[str] = ".*") -> None:
        if pattern is not None:
            pattern = re.compile(pattern)

        for name, parameter in self._model.base_model.named_parameters():
            if pattern is not None and pattern.match(name):
                parameter.requires_grad = False


class HuggingfaceLanguageModel(HuggingfaceModelBase):
    @classmethod
    def from_path(cls, path: str) -> "HuggingfaceLanguageModel":
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0
        return cls(model, tokenizer, pad_token_id)

    def freeze(self, pattern: Optional[str] = ".*") -> None:
        super().freeze(pattern)

        for parameter in self._model.lm_head.parameters():
            parameter.requires_grad = True

    def forward(self, x: PackedSequence) -> PackedSequence:
        x, lengths = unpack_to_padded(x, pad_value=self._pad_token_id)
        x = x[:, :, 0]
        mask = make_mask(x, lengths)
        y = self._model(x, attention_mask=mask)
        return pack_padded(y.logits.log_softmax(-1), lengths)


class HuggingfaceRegressionModel(HuggingfaceModelBase):
    @classmethod
    def from_path(cls, path: str) -> "HuggingfaceRegressionModel":
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            pad_token_id = tokenizer.eos_token_id
        else:
            pad_token_id = 0
        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=1,
            problem_type="regression",
            pad_token_id=pad_token_id,
        )
        return cls(model, tokenizer, pad_token_id)

    def freeze(self, pattern: Optional[str] = ".*") -> None:
        super().freeze(pattern)

        for parameter in self._model.score.parameters():
            parameter.requires_grad = True

    def forward(self, x: PackedSequence) -> Tensor:
        x, lengths = unpack_to_padded(x, pad_value=self._pad_token_id)
        x = x[:, :, 0]
        mask = make_mask(x, lengths)
        return self._model(x, attention_mask=mask).logits