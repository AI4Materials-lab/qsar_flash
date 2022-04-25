import flash
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torchmetrics import MeanAbsoluteError, R2Score

from qsar_flash import FolderDataset, GraphRegressionData, MolecularGraphRegressor


def main():
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

    # 2. Build the task
    backbone_kwargs = dict(
        hidden_channels=256,
        num_filters=256,
        num_interactions=6,
        num_gaussians=100,
        cutoff=10.0,
        max_num_neighbors=32,
    )

    epochs = 300
    lr = 0.0001

    lr_scheduler_kwargs = dict(max_lr=lr, total_steps=len(datamodule.train_dataloader()) * epochs)  # type: ignore

    lr_scheduler_pl_kwargs = dict(interval="step")
    model = MolecularGraphRegressor(
        backbone="SchNet",
        metrics=[MeanAbsoluteError(), R2Score()],
        learning_rate=lr,
        pooling_fn="add",
        optimizer="AdamW",
        loss_fn=F.l1_loss,
        lr_scheduler=("onecyclelr", lr_scheduler_kwargs, lr_scheduler_pl_kwargs),
        backbone_kwargs=backbone_kwargs,
    )

    wandb_logger = WandbLogger(
        name="schnet-tmqm",
        project="graph-drug",
        entity="inno-materials-ai",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 3. Create the trainer and fit the model
    trainer = flash.Trainer(max_epochs=epochs, gpus=[2], logger=wandb_logger, callbacks=[lr_monitor])
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
