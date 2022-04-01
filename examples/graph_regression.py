import flash
from flash.core.utilities.imports import example_requires
from torchmetrics import MeanAbsoluteError

from qsar_flash import GraphRegressionData, GraphRegressor
from qsar_flash.datasets.qm9_property import QM9Property

example_requires("graph")


def main() -> None:
    # 1. Create the DataModule
    dataset = QM9Property(property_index=4, root="data/qm9")

    datamodule = GraphRegressionData.from_datasets(
        train_dataset=dataset,
        val_split=0.2,
        batch_size=64,
        num_workers=64,
        pin_memory=True,
    )

    # 2. Build the task
    backbone_kwargs = {"hidden_channels": 512, "num_layers": 4}
    model = GraphRegressor(
        backbone="GCN",
        metrics=MeanAbsoluteError(),
        learning_rate=0.001,
        pooling_fn="add",
        optimizer="AdamW",
        num_features=datamodule.num_features,
        backbone_kwargs=backbone_kwargs,
    )

    # 3. Create the trainer and fit the model
    trainer = flash.Trainer(max_epochs=1, gpus=[0])
    trainer.fit(model, datamodule=datamodule)

    # 4. Regressify some graphs!
    datamodule = GraphRegressionData.from_datasets(
        predict_dataset=dataset[:100],
        batch_size=64,
    )
    predictions = trainer.predict(model, datamodule=datamodule)
    print(predictions)

    # 5. Save the model!
    # trainer.save_checkpoint("graph_regression_model.pt")


if __name__ == "__main__":
    main()
