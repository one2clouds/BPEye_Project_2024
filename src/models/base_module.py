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
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


class BaseModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        root_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler.LinearLR] = None,
        scheduler_monitor: str = "val/loss",
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


    def forward(self, x, classes):
        return self.net(x, num_classes=len(classes))
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # print(batch[0])
        # print("-"*80)
        # print(batch[1])
        # print("-"*80)
        # print(batch[2])
        # print("-"*80)
        train_path = os.path.join(self.root_dir, "train")
        root=pathlib.Path(train_path)
        classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])

        x, y = batch[0], batch[1]
        # print(x)
        # print(y) # tensor([1, 1, 1, 1], device='cuda:0')


        # print(x.shape) # torch.Size([4, 1, 128, 128, 128])
        # print(y.shape) # torch.Size([4, 1, 128, 128, 128])
        # print(x.max()) # tensor(9.2463) # because of normalize intensity
        # print(x.min()) # tensor(-7.0969)
        # print(y.unique()) # tensor([0., 1., 2., 3.])

        logits = self.forward(x, classes)
        softmax_logits = nn.Softmax(dim=1)(logits)

        # print(softmax_logits.max()) #tensor(0.9980, device='cuda:0')
        # print(softmax_logits.min()) #tensor(3.7885e-07, device='cuda:0')
        # print(softmax_logits.shape) #torch.Size([4, 5, 128, 128, 128])
        # print(y.shape) # torch.Size([4, 1, 128, 128, 128])

        loss = self.loss_fn(softmax_logits, y)
        y_pred = torch.argmax(softmax_logits, dim=1)

        # print(y_pred) # tensor([1, 1, 1, 1], device='cuda:0')
        # print(y) # tensor([1, 1, 1, 1], device='cuda:0')

        accuracy = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        precision = precision_score(y.cpu().numpy(), y_pred.cpu().numpy())
        recall = recall_score(y.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y.cpu().numpy(), y_pred.cpu().numpy())

        return loss, y, y_pred, x, [accuracy, precision, recall, f1]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        loss, targets, preds,  x, metric_list = self.model_step(batch)
        
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"train/acc" : metric_list[0], "train/precision" : metric_list[1], "train/recall" : metric_list[2], "train/f1" : metric_list[3]})

        # if isinstance(self.logger, WandbLogger):
        #     a_3d_img_to_tif(original_img, 'original_img.gif')
        #     self.logger.log_video(key="train/original_img", videos=['original_img.gif'])
        #     a_3d_img_to_tif(targets_arg, 'target_img.gif', segmentation=True)
        #     self.logger.log_video(key="train/target_img", videos=['target_img.gif'])
        #     a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
        #     self.logger.log_video(key="train/pred_img", videos=['predicted_img.gif'])

        # return loss or backpropagation will fail
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, targets, preds,  x, metric_list = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss.compute().item(), on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"val/acc" : metric_list[0], "val/precision" : metric_list[1], "val/recall" : metric_list[2], "val/f1" : metric_list[3]})

        # if isinstance(self.logger, WandbLogger):
        #     a_3d_img_to_tif(original_img, 'original_img.gif')
        #     self.logger.log_video(key="val/original_img", videos=['original_img.gif'])
        #     a_3d_img_to_tif(targets_arg, 'target_img.gif', segmentation=True)
        #     self.logger.log_video(key="val/target_img", videos=['target_img.gif'])
        #     a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
        #     self.logger.log_video(key="val/pred_img", videos=['predicted_img.gif'])

    def on_validation_epoch_end(self) -> None:
        loss = self.val_loss.compute() # gives current loss from batch 
        self.val_loss_min(loss) # logs our loss into the min 

        # self.log("val/loss", self.val_loss_min(loss).item(), sync_dist=True, prog_bar=True) # get the lowest value from the above


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, targets, preds,  x, metric_list = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict({"test/acc" : metric_list[0], "test/precision" : metric_list[1], "test/recall" : metric_list[2], "test/f1" : metric_list[3]})

        # if isinstance(self.logger, WandbLogger):
        #     a_3d_img_to_tif(original_img, 'original_img.gif')
        #     self.logger.log_video(key="test/original_img", videos=['original_img.gif'])
        #     a_3d_img_to_tif(targets_arg,'target_img.gif', segmentation=True )
        #     self.logger.log_video(key="test/target_img", videos=['target_img.gif'])
        #     a_3d_img_to_tif(pred_arg, 'predicted_img.gif', segmentation=True)
        #     self.logger.log_video(key="test/pred_img", videos=['predicted_img.gif'])


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