import flash
from flash.core.utilities.imports import example_requires
from flash.graph import GraphClassificationData, GraphClassifier

example_requires("graph")

from torch_geometric.datasets import TUDataset  # noqa: E402


def main() -> None:
    # 1. Create the DataModule
    dataset = TUDataset(root="data", name="KKI")

    datamodule = GraphClassificationData.from_datasets(
        train_dataset=dataset,
        val_split=0.1,
        batch_size=4,
    )

    # 2. Build the task
    backbone_kwargs = {"hidden_channels": 512, "num_layers": 4}
    model = GraphClassifier(
        num_features=datamodule.num_features, num_classes=datamodule.num_classes, backbone_kwargs=backbone_kwargs
    )

    # 3. Create the trainer and fit the model
    trainer = flash.Trainer(max_epochs=3, gpus=1)
    trainer.fit(model, datamodule=datamodule)

    # 4. Classify some graphs!
    datamodule = GraphClassificationData.from_datasets(
        predict_dataset=dataset[:100],
        batch_size=4,
    )
    predictions = trainer.predict(model, datamodule=datamodule, output="classes")
    print(predictions)

    # 5. Save the model!
    # trainer.save_checkpoint("graph_classification_model.pt")


if __name__ == "__main__":
    main()
