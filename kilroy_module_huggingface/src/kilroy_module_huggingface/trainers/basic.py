from dataclasses import dataclass
from typing import Any, AsyncIterable, Dict, List, Set, Tuple

import numpy as np
import torch
from aiostream import stream
from kilroy_module_pytorch_py_sdk import (
    pack_list,
    truncate_first_element,
    truncate_last_element,
)
from kilroy_module_server_py_sdk import (
    Configurable,
    Metric,
    SerializableModel,
    background,
    classproperty,
)
from torch import Tensor
from torch.nn import NLLLoss

from kilroy_module_huggingface.models import HuggingfaceLanguageModel
from kilroy_module_huggingface.optimizers import Optimizer
from kilroy_module_huggingface.trainers.base import Trainer


class SupervisedLossMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "supervisedLoss"

    @classproperty
    def label(cls) -> str:
        return "Supervised Loss"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "loss"}},
        }


class ReinforcedScoreMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "reinforcedScore"

    @classproperty
    def label(cls) -> str:
        return "Reinforced Score"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "score"}},
        }


class Params(SerializableModel):
    batch_size: int


@dataclass
class State:
    batch_size: int
    epoch: int
    supervised_loss_metric: SupervisedLossMetric
    reinforced_score_metric: ReinforcedScoreMetric
    epoch_supervised_losses: List[float]
    epoch_reinforced_scores: List[float]


class BasicTrainer(Trainer, Configurable[State]):
    async def build_default_state(self) -> State:
        params = Params.parse_obj(self._kwargs)
        return State(
            batch_size=params.batch_size,
            epoch=0,
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            epoch_supervised_losses=[],
            epoch_reinforced_scores=[],
        )

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return {
                state.supervised_loss_metric,
                state.reinforced_score_metric,
            }

    async def fit_supervised(
        self, model: HuggingfaceLanguageModel, data: AsyncIterable[Tensor]
    ) -> None:
        # noinspection PyShadowingNames
        def fit(batch):
            input = pack_list(truncate_last_element(batch))
            target = pack_list(truncate_first_element(batch))
            logprobs = model(input)
            loss = NLLLoss()(logprobs.data, target.data.flatten())
            loss.backward()
            return loss.item()

        async with self.state.read_lock() as state:
            batches = stream.chunks(data, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                async with self.state.write_lock():
                    loss = await background(fit, batch)
                    state.epoch_supervised_losses.append(loss)

    async def fit_reinforced(
        self,
        model: HuggingfaceLanguageModel,
        results: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        results = list([result async for result in results])
        logprobs = torch.stack([logprob for _, logprob, _ in results])
        scores = torch.stack([score for _, _, score in results])

        def fit():
            loss = -(logprobs * scores).mean()
            loss.backward()
            return scores.mean().item()

        async with self.state.write_lock() as state:
            score = await background(fit)
            state.epoch_reinforced_scores.append(score)

    async def step(self, optimizer: Optimizer) -> None:
        async with self.state.write_lock() as state:
            await optimizer.step()
            if state.epoch_supervised_losses:
                await state.supervised_loss_metric.report(
                    {
                        "epoch": state.epoch,
                        "loss": np.mean(state.epoch_supervised_losses),
                    }
                )
            if state.epoch_reinforced_scores:
                await state.reinforced_score_metric.report(
                    {
                        "epoch": state.epoch,
                        "score": np.mean(state.epoch_reinforced_scores),
                    }
                )
            state.epoch_supervised_losses = []
            state.epoch_reinforced_scores = []
            state.epoch += 1
