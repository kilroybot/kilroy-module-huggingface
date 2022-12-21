import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoModelForTokenClassification,
)

from kilroy_module_huggingface.utils import suppress_huggingface_logging
from kilroy_module_pytorch_py_sdk import (
    SequentialModel,
    Savable,
    background,
    pack_padded,
    unpack_to_padded,
)

BaseType = TypeVar("BaseType", bound="HuggingfaceSequentialModelBase")


def make_mask(x: Tensor, lengths: Tensor) -> Tensor:
    indices = torch.arange(x.shape[-1]).repeat((len(x), 1))
    lengths = lengths.view(-1, 1)
    return indices < lengths


class HuggingfaceSequentialModelBase(SequentialModel, Savable, ABC):
    def __init__(self, model: PreTrainedModel, pad_token_id: int) -> None:
        super().__init__()
        self._model = model
        self._pad_token_id = pad_token_id

    @classmethod
    @abstractmethod
    async def from_path(cls: Type[BaseType], path: str) -> BaseType:
        pass

    async def save(self, directory: Path) -> None:
        model_directory = directory / "model"
        config_path = directory / "config.json"
        model_directory.mkdir(parents=True, exist_ok=True)

        await background(self._model.save_pretrained, model_directory)
        with open(config_path, "w") as f:
            json.dump({"pad_token_id": self._pad_token_id}, f)

    @classmethod
    async def from_saved(
        cls: Type[BaseType], directory: Path, **kwargs
    ) -> BaseType:
        model_directory = directory / "model"
        config_path = directory / "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        model = await background(cls._load_saved_model, model_directory)
        return cls(model, config["pad_token_id"])

    @classmethod
    @abstractmethod
    def _load_saved_model(cls, directory: Path) -> PreTrainedModel:
        pass

    @property
    def base_model(self) -> nn.Module:
        return self._model.base_model

    def freeze(self, pattern: str = ".*") -> None:
        pattern = re.compile(pattern)

        for name, parameter in self._model.base_model.named_parameters():
            if pattern.match(name):
                parameter.requires_grad = False


class HuggingfaceLanguageModel(HuggingfaceSequentialModelBase):
    @classmethod
    def _load_saved_model(cls, directory: Path) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(directory)

    @classmethod
    async def from_path(cls: Type[BaseType], path: str) -> BaseType:
        model = await background(AutoModelForCausalLM.from_pretrained, path)
        if model.config.pad_token_id is not None:
            pad_token_id = model.config.pad_token_id
        elif model.config.eos_token_id is not None:
            pad_token_id = model.config.eos_token_id
        else:
            pad_token_id = 0
        return cls(model, pad_token_id)

    def freeze(self, pattern: str = ".*") -> None:
        super().freeze(pattern)

        for parameter in self._model.lm_head.parameters():
            parameter.requires_grad = True

    def forward(self, x: PackedSequence) -> PackedSequence:
        x, lengths = unpack_to_padded(x, pad_value=self._pad_token_id)
        x = x[:, :, 0]
        mask = make_mask(x, lengths)
        y = self._model(x, attention_mask=mask)
        return pack_padded(y.logits.log_softmax(-1), lengths)


class HuggingfaceValueModel(HuggingfaceSequentialModelBase):
    @classmethod
    def _load_saved_model(cls, directory: Path) -> PreTrainedModel:
        return AutoModelForTokenClassification.from_pretrained(directory)

    @classmethod
    async def from_path(cls: Type[BaseType], path: str) -> BaseType:
        config = await background(AutoConfig.from_pretrained, path)
        if config.pad_token_id is not None:
            pad_token_id = config.pad_token_id
        elif config.eos_token_id is not None:
            pad_token_id = config.eos_token_id
        else:
            pad_token_id = 0
        with suppress_huggingface_logging():
            model = await background(
                AutoModelForTokenClassification.from_pretrained,
                path,
                num_labels=1,
                problem_type="regression",
                pad_token_id=pad_token_id,
                ignore_mismatched_sizes=True,
            )
        return cls(model, pad_token_id)

    def freeze(self, pattern: str = ".*") -> None:
        super().freeze(pattern)

        for parameter in self._model.classifier.parameters():
            parameter.requires_grad = True

    def forward(self, x: PackedSequence) -> PackedSequence:
        x, lengths = unpack_to_padded(x, pad_value=self._pad_token_id)
        x = x[:, :, 0]
        mask = make_mask(x, lengths)
        y = self._model(x, attention_mask=mask)
        return pack_padded(y.logits, lengths)
