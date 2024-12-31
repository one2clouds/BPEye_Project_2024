import re
from typing import Any, Callable, Tuple, Optional

import torch
# from pytorch_lightning import LightningModule
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric, MinMetric
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import Compose, ToTensor, ToPILImage
import pathlib
import os
from sklearn.metrics import balanced_accuracy_score, precision_score, f1_score, recall_score


class BaseModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        root_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None,
        scheduler_monitor: str = "val/f1",
        ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.net = net
        # loss function
        self.loss_fn = loss_fn

        self.root_dir = root_dir

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_loss_min = MinMetric()

        self.accuracy = balanced_accuracy_score
        self.precision = precision_score
        self.recall = recall_score
        self.f1 = f1_score


    def forward(self, x, classes):
        return self.net(x, num_classes=len(classes))
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        train_path = os.path.join(self.root_dir, "train")
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

        x, y = batch[0], batch[1]
        logits = self.forward(x, classes)

        # print(logits)
        # print(y)
        # print(logits)

        loss = self.loss_fn(logits, y)

        # print(sigmoid_logits)
        # print(y)

        # sigmoid_logits = nn.Sigmoid()(logits)
        preds = torch.argmax(logits, dim=1)

        # print(f"this is preds : {preds}") # tensor([1, 1, 1, 1], device='cuda:0')
        # print(f"this is y : {y}") # tensor([1, 1, 1, 1], device='cuda:0')

        accuracy = self.accuracy(y.cpu().numpy(), preds.cpu().numpy())
        precision = self.precision(y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0.0)
        recall = self.recall(y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0.0)
        f1 = self.f1(y.cpu().numpy(), preds.cpu().numpy(), average='weighted', zero_division=0.0)

        return loss, preds, y, [accuracy, precision, recall, f1]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, preds, y, metric_list = self.model_step(batch)

        self.train_loss(loss)

        # print("Training")
        # print(f"this is preds : {preds}") # tensor([1, 1, 1, 1], device='cuda:0')
        # print(f"this is y : {y}") # tensor([1, 1, 1, 1], device='cuda:0')
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", metric_list[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/precision", metric_list[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", metric_list[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", metric_list[3], on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        pass
        

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, y, metric_list = self.model_step(batch)

        # print("validation")
        # print(f"this is preds : {preds}") # tensor([1, 1, 1, 1], device='cuda:0')
        # print(f"this is y : {y}") # tensor([1, 1, 1, 1], device='cuda:0')
        
        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", metric_list[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/precision", metric_list[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recall", metric_list[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", metric_list[3], on_step=False, on_epoch=True, prog_bar=True)
        return None

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, y, metric_list = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", metric_list[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", metric_list[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", metric_list[2], on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", metric_list[3], on_step=False, on_epoch=True, prog_bar=True)
        return None

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.scheduler_monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "mnist.yaml")
    _ = hydra.utils.instantiate(cfg)