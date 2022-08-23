import asyncio
from asyncio import Queue, Task
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    Coroutine,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import torch
from aiostream import stream
from aiostream.aiter_utils import aiter, anext
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
from torch.nn import MSELoss, NLLLoss
from torch.nn.utils.rnn import PackedSequence

from kilroy_module_huggingface.generator import Generator
from kilroy_module_huggingface.models import (
    HuggingfaceLanguageModel,
    HuggingfaceRegressionModel,
)
from kilroy_module_huggingface.optimizers import Optimizer
from kilroy_module_huggingface.trainers.base import Trainer
from kilroy_module_huggingface.utils import freeze


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


class RewardModelLossMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "rewardModelLoss"

    @classproperty
    def label(cls) -> str:
        return "Reward Model Loss"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "loss"}},
        }


class RewardModelScoreMetric(Metric[Dict]):
    @classproperty
    def name(cls) -> str:
        return "rewardModelScore"

    @classproperty
    def label(cls) -> str:
        return "Reward Model Score"

    @classproperty
    def config(cls) -> Dict[str, Any]:
        return {
            "type": "line",
            "data": {"datasets": [{"data": []}]},
            "options": {"parsing": {"xAxisKey": "epoch", "yAxisKey": "score"}},
        }


class Params(SerializableModel):
    model_name: str
    freeze: Optional[str] = None
    optimizer_type: str = "adam"
    optimizers_params: Dict[str, Dict[str, Any]] = {}
    generator_params: Dict[str, Any] = {}
    batch_size: int
    sample_size: int


@dataclass
class State:
    model: HuggingfaceRegressionModel
    optimizer: Optimizer
    optimizers_params: Dict[str, Dict[str, Any]]
    generator: Generator
    batch_size: int
    sample_size: int
    epoch: int
    supervised_loss_metric: SupervisedLossMetric
    reinforced_score_metric: ReinforcedScoreMetric
    reward_model_loss_metric: RewardModelLossMetric
    reward_model_score_metric: RewardModelScoreMetric
    epoch_supervised_losses: List[float]
    epoch_reinforced_scores: List[float]
    epoch_reward_model_losses: List[float]
    epoch_reward_model_scores: List[float]
    coroutine_queue: Queue[Coroutine]
    worker_task: Task


class RewardModelTrainer(Trainer, Configurable[State]):
    @staticmethod
    async def _build_model(
        params: Params,
    ) -> HuggingfaceRegressionModel:
        return await background(
            HuggingfaceRegressionModel.from_path, params.model_name
        )

    @staticmethod
    async def _build_optimizer(
        params: Params, model: HuggingfaceRegressionModel
    ) -> Optimizer:
        opt_cls = Optimizer.for_category(params.optimizer_type)
        opt_params = params.optimizers_params.get(params.optimizer_type, {})
        if issubclass(opt_cls, Configurable):
            optimizer = await opt_cls.build(
                params=model.parameters(), **opt_params
            )
            await optimizer.init()
        else:
            optimizer = opt_cls(model.parameters(), **opt_params)
        return optimizer

    @staticmethod
    async def _build_generator(params: Params) -> Generator:
        generator = await Generator.build(**params.generator_params)
        await generator.init()
        return generator

    async def build_default_state(self) -> State:
        params = Params.parse_obj(self._kwargs)
        model = await self._build_model(params)
        optimizer = await self._build_optimizer(params, model)
        model.freeze(params.freeze)
        queue = Queue()
        return State(
            model=model,
            optimizer=optimizer,
            optimizers_params=params.optimizers_params,
            generator=await self._build_generator(params),
            batch_size=params.batch_size,
            sample_size=params.sample_size,
            epoch=0,
            supervised_loss_metric=await SupervisedLossMetric.build(),
            reinforced_score_metric=await ReinforcedScoreMetric.build(),
            reward_model_loss_metric=await RewardModelLossMetric.build(),
            reward_model_score_metric=await RewardModelScoreMetric.build(),
            epoch_supervised_losses=[],
            epoch_reinforced_scores=[],
            epoch_reward_model_losses=[],
            epoch_reward_model_scores=[],
            coroutine_queue=queue,
            worker_task=asyncio.create_task(self._work(queue)),
        )

    @staticmethod
    async def _work(queue: Queue[Coroutine]) -> None:
        while True:
            coroutine = await queue.get()
            await coroutine

    async def cleanup(self) -> None:
        async with self.state.write_lock() as state:
            state.worker_task.cancel()
            try:
                await state.worker_task
            except asyncio.CancelledError:
                pass
            for _ in range(state.coroutine_queue.qsize()):
                coroutine = state.coroutine_queue.get_nowait()
                coroutine.close()
                state.coroutine_queue.task_done()

    async def get_metrics(self) -> Set[Metric]:
        async with self.state.read_lock() as state:
            return {
                state.supervised_loss_metric,
                state.reinforced_score_metric,
                state.reward_model_loss_metric,
                state.reward_model_score_metric,
            }

    @staticmethod
    def _fit_supervised_batch(
        model: HuggingfaceLanguageModel, batch: Iterable[Tensor]
    ) -> float:
        input = pack_list(truncate_last_element(batch))
        target = pack_list(truncate_first_element(batch))
        logprobs = model(input)
        loss = NLLLoss()(logprobs.data, target.data.flatten())
        loss.backward()
        return loss.item()

    async def fit_supervised(
        self, model: HuggingfaceLanguageModel, data: AsyncIterable[Tensor]
    ) -> None:
        async with self.state.read_lock() as state:
            batches = stream.chunks(data, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                async with self.state.write_lock():
                    loss = await background(self._fit_supervised_batch, batch)
                    state.epoch_supervised_losses.append(loss)

    @staticmethod
    def _fit_reward_model_batch(
        model: HuggingfaceRegressionModel,
        sequences: PackedSequence,
        scores: Tensor,
    ) -> float:
        predicted = model(sequences)
        loss = MSELoss()(predicted, scores)
        loss.backward()
        return loss.item()

    async def _fit_with_reward_model(
        self, model: HuggingfaceLanguageModel
    ) -> None:
        # noinspection PyShadowingNames
        def fit(model, batch):
            with freeze(model) as frozen:
                scores = frozen(batch.sequences)
            loss = -(batch.logprobs * scores).mean()
            loss.backward()
            return scores.mean().item()

        async with self.state.read_lock() as state:
            generated = state.generator.generate(model, state.sample_size)

        generated = aiter(generated)

        while True:
            async with self.state.write_lock() as state:
                try:
                    batch = await anext(generated)
                except StopAsyncIteration:
                    break
                score = await background(fit, state.model, batch)
                state.epoch_reward_model_scores.append(score)

    async def fit_reinforced(
        self,
        model: HuggingfaceLanguageModel,
        results: AsyncIterable[Tuple[Tensor, Tensor, Tensor]],
    ) -> None:
        async with self.state.read_lock() as state:
            batches = stream.chunks(results, state.batch_size)

        async with batches.stream() as streamer:
            async for batch in streamer:
                sequences = pack_list([sequence for sequence, _, _ in batch])
                scores = torch.vstack([score for _, _, score in batch])
                async with self.state.write_lock() as state:
                    loss = await background(
                        self._fit_reward_model_batch,
                        state.model,
                        sequences,
                        scores,
                    )
                    state.epoch_reward_model_losses.append(loss)
                    state.epoch_reinforced_scores.append(scores.mean().item())

        async with self.state.write_lock() as state:
            await state.coroutine_queue.put(self._fit_with_reward_model(model))

    async def step(self, optimizer: Optimizer) -> None:
        async with self.state.write_lock() as state:
            await state.optimizer.step()
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
            if state.epoch_reward_model_losses:
                await state.reward_model_loss_metric.report(
                    {
                        "epoch": state.epoch,
                        "loss": np.mean(state.epoch_reward_model_losses),
                    }
                )
            if state.epoch_reward_model_scores:
                await state.reward_model_score_metric.report(
                    {
                        "epoch": state.epoch,
                        "score": np.mean(state.epoch_reward_model_scores),
                    }
                )
            state.epoch_supervised_losses = []
            state.epoch_reinforced_scores = []
            state.epoch_reward_model_losses = []
            state.epoch_reward_model_scores = []
            state.epoch += 1
