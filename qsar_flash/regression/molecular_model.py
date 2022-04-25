from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from flash.core.regression import RegressionTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.types import LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
from flash.core.model import OutputKeys

from qsar_flash.regression.model import POOLING_FUNCTIONS


from qsar_flash.backbones import MOLECULAR_GRAPH_BACKBONES


class MolecularData(Protocol):
    z: torch.Tensor
    pos: torch.Tensor
    batch: Optional[torch.Tensor]


class MolecularGraphRegressor(RegressionTask):
    """The ``GraphRegressor`` is a :class:`~flash.Task` for regressing graphs.

    Args:
        num_features (int): The number of features in the input.
        backbone: Name of the backbone to use.
        backbone_kwargs: Dictionary dependent on the backbone, containing for example in_channels, out_channels,
            hidden_channels or depth (number of layers).
        pooling_fn: The global pooling operation to use (one of: "mean", "max", "add" or a callable).
        head: The head to use.
        loss_fn: Loss function for training, defaults to mean squared error.
        learning_rate: Learning rate to use for training.
        optimizer: Optimizer to use for training.
        lr_scheduler: The LR scheduler to use during training.
        metrics: Metrics to compute for training and evaluation.
    """

    backbones: FlashRegistry = MOLECULAR_GRAPH_BACKBONES

    required_extras: str = "graph"

    def __init__(
        self,
        backbone: Union[str, Tuple[nn.Module, int]] = "SchNet",
        backbone_kwargs: Optional[Dict] = None,
        pooling_fn: Union[str, Callable] = "mean",
        head: Optional[Union[Callable, nn.Module]] = None,
        loss_fn: LOSS_FN_TYPE = F.mse_loss,
        learning_rate: Optional[float] = None,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        self.save_hyperparameters(ignore=["metrics", "head"])

        super().__init__(
            loss_fn=loss_fn,  # type: ignore
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.mean = mean
        self.std = std

        if isinstance(backbone, tuple):
            self.backbone, num_out_features = backbone
        else:
            backbone_kwargs = {} if backbone_kwargs is None else backbone_kwargs
            self.backbone, num_out_features = self.backbones.get(backbone)(**backbone_kwargs)  # type: ignore

        self.pooling_fn = POOLING_FUNCTIONS[pooling_fn] if isinstance(pooling_fn, str) else pooling_fn

        if head is not None:
            self.head = head
        else:
            self.head = nn.Identity(num_out_features)

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        target = self.prepare_target(batch[DataKeys.TARGET])
        batch = (batch[DataKeys.INPUT], target)
        output = self.step(batch, batch_idx, self.train_metrics)
        self.log_dict(
            {f"train/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[1].size(1),
        )
        return output[OutputKeys.LOSS]

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        target = self.prepare_target(batch[DataKeys.TARGET])
        batch = (batch[DataKeys.INPUT], target)
        output = self.step(batch, batch_idx, self.val_metrics)
        self.log_dict(
            {f"val/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[1].size(1),
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        target = self.prepare_target(batch[DataKeys.TARGET])
        batch = (batch[DataKeys.INPUT], target)
        output = self.step(batch, batch_idx, self.test_metrics)
        self.log_dict(
            {f"test/{k}": v for k, v in output[OutputKeys.LOGS].items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch[1].size(1),
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return super().predict_step(batch[DataKeys.INPUT], batch_idx, dataloader_idx=dataloader_idx)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def prepare_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            return self.normalize(x)
        return x

    def to_metrics_format(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            return x * self.std + self.mean
        return x

    def forward(self, data: MolecularData) -> torch.Tensor:
        x = self.backbone(data.z, data.pos, data.batch)
        x = self.pooling_fn(x, data.batch)
        return self.head(x)
