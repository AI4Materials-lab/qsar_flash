from typing import Tuple

import flash
import numpy as np

# from clearml import Task
from flash.core.data.io.input import DataKeys
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, R2Score
from tqdm import tqdm

from qsar_flash import FolderDataset, GraphRegressionData, MolecularGraphRegressor


def calculate_mean_std(dataset) -> Tuple[float, float]:
    y = np.array([dataset[i][DataKeys.TARGET].item() for i in tqdm(range(len(dataset)))])
    return float(y.mean()), float(y.std())


def main():
    name = "edgeupdatenet-tmqm"
    # task = Task.init(
    #     project_name="GraphDrugs",
    #     task_name=name,
    #     output_uri="s3://api.blackhole.ai.innopolis.university:443/new-material-clearml/",
    # )
    seed_everything(42)
    dataset = FolderDataset.read_csv(
        "/mnt/storage/new_materials/data/tmqm/xyz/xyz_molecules/",
        "id_prop.csv",
        name="tmqm",
        filename_column_name="filename",
        target_column_name="target",
    )
    datamodule = GraphRegressionData.from_datasets(
        train_dataset=dataset,
        val_split=0.2,
        batch_size=64,
        num_workers=64,
        pin_memory=True,
    )

    # mean, std = calculate_mean_std(datamodule.train_dataset)
    # print(f"{mean=} {std=}")

    # 2. Build the task
    backbone_kwargs = dict(hidden_channels=256, num_interactions=6, num_gaussians=100, cutoff=10.0, out_channels=1)

    epochs = 300
    lr = 0.0001

    lr_scheduler_kwargs = dict(max_lr=lr, total_steps=len(datamodule.train_dataloader()) * epochs)  # type: ignore

    lr_scheduler_pl_kwargs = dict(interval="step")

    model = MolecularGraphRegressor(
        backbone="EdgeUpdateNet",
        metrics=[MeanAbsoluteError(), R2Score()],
        learning_rate=lr,
        pooling_fn="add",
        optimizer="AdamW",
        loss_fn=F.l1_loss,
        lr_scheduler=("onecyclelr", lr_scheduler_kwargs, lr_scheduler_pl_kwargs),
        backbone_kwargs=backbone_kwargs,
        # mean=mean,
        # std=std,
    )

    wandb_logger = WandbLogger(
        name=name,
        project="graph-drug",
        entity="inno-materials-ai",
    )
    tensorboard_logger = TensorBoardLogger(name=name, save_dir="./tb_logs")
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckeckpoint = ModelCheckpoint(
        monitor="val/l1_loss",
    )

    # 3. Create the trainer and fit the model
    trainer = flash.Trainer(
        max_epochs=epochs,
        gpus=[2],
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[lr_monitor, model_ckeckpoint],
        gradient_clip_val=10,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
