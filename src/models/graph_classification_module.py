from typing import Any, List
from xml.etree.ElementTree import TreeBuilder

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import MaxMetric

from .util import get_classifiation_metric


class GraphClassificationModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module = None,
        datamodule: LightningDataModule = None,
        pool_method: str = "add",
        output_head=True,
        num_tasks=None,
        lr: float = 0.001,
        weight_decay: float = 0,
        criterion=None,
        **kwargs,
    ):
        super().__init__()

        self.model = model

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        metric_cls, self.metric_name = get_classifiation_metric(
            self.hparams.datamodule.dataset
        )

        self.train_metric = metric_cls()
        self.val_metric = metric_cls()
        self.test_metric = metric_cls()
        self.val_metric_best = MaxMetric()

        if pool_method == "add":
            self.pool_fn = lambda x: x.sum(1)
        elif pool_method is None:
            self.pool_fn = lambda x: x

        if output_head is True:
            self.output_head = nn.Linear(self.model.hidden_dim, num_tasks)

        if criterion == "bce":
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):

        out = self.hparams.model(batch)

        if self.hparams.pool_method is not None:
            preds = self.pool_fn(out)
        else:
            preds = out

        if self.hparams.output_head:
            preds = self.output_head(preds)

        loss = self.loss_fn(preds, batch.y)

        return loss, preds, batch.y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        metric = self.train_metric((F.sigmoid(preds) > 0.5).float(), targets.long())
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            f"train/{self.metric_name}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        metric = self.val_metric((F.sigmoid(preds) > 0.5).float(), targets.long())
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            f"val/{self.metric_name}",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        metric = self.val_metric.compute()  # get val accuracy from current epoch
        self.val_metric_best.update(metric)
        self.log(
            f"val/{self.metric_name}_best",
            self.val_metric_best.compute(),
            on_epoch=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        metric = self.test_metric(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log(f"test/{self.metric_name}", metric, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_metric.reset()
        self.test_metric.reset()
        self.val_metric.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
