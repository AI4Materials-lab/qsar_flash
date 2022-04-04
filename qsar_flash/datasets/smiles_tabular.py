from pathlib import Path
from typing import Any, Callable, cast, Dict, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from qsar_flash.utils import smiles2graph


class MolecularData(Protocol):
    z: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor
    __num_nodes__: int


class SmilesTabular(InMemoryDataset):
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        name: str,
        smiles_column_name: str = "smiles",
        target_column_name: str = "target",
        root: str = "data",
        smiles2graph: Callable[[str], Dict[str, Any]] = smiles2graph,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.df = df
        self.smiles_column_name = smiles_column_name
        self.target_column_name = target_column_name
        self.smiles2graph = smiles2graph
        self.original_root = Path(root)
        self.folder = self.original_root / name

        super().__init__(str(self.folder), transform, pre_transform, pre_filter)

        self.data, self.slices = self.load_processed_data()
        self.data.z = self.data.z.long()

    @classmethod
    def read_csv(
        cls,
        reader: str,
        *,
        name: str,
        smiles_column_name: str = "smiles",
        target_column_name: str = "target",
        root: str = "data",
        smiles2graph: Callable[[str], Dict[str, Any]] = smiles2graph,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **pandas_kwargs,
    ):

        df = cast(pd.DataFrame, pd.read_csv(reader, usecols=(smiles_column_name, target_column_name), **pandas_kwargs))
        return cls(
            df=df,
            name=name,
            root=root,
            smiles2graph=smiles2graph,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @classmethod
    def read_df(
        cls,
        df: pd.DataFrame,
        *,
        name: str,
        smiles_column_name: str = "smiles",
        target_column_name: str = "target",
        root: str = "data",
        smiles2graph: Callable[[str], Dict[str, Any]] = smiles2graph,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        df = df.loc[:, [smiles_column_name, target_column_name]]
        return cls(
            df=df,
            name=name,
            root=root,
            smiles2graph=smiles2graph,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @property
    def processed_file_names(self) -> str:
        return "geometric_data_preprocessed.pt"

    def _download(self) -> None:
        pass

    def process(self):
        # Read data into huge `Data` list.
        df = self.df.drop_duplicates(subset=self.smiles_column_name)

        smiles_list = df[self.smiles_column_name].values
        target_list = df[self.target_column_name].values

        data_list = []
        for i in tqdm(range(len(smiles_list)), desc="Convering SMILES to Data"):
            data: MolecularData = cast(MolecularData, Data())

            smiles = smiles_list[i]
            target = target_list[i]
            try:
                graph = self.smiles2graph(smiles)
            except ValueError as e:
                print(e)
                continue

            data.__num_nodes__ = int(graph["num_nodes"])
            data.z = torch.from_numpy(graph["z"]).to(torch.int8)
            data.y = torch.Tensor([target])
            data.pos = torch.tensor(graph["pos"]).to(torch.float32)

            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)  # type: ignore
        torch.save((data, slices), self.processed_paths[0])

    def load_processed_data(self) -> Tuple[MolecularData, Optional[Dict[str, torch.Tensor]]]:
        return torch.load(self.processed_paths[0])

    def __getitem__(self, idx: Union[int, np.integer]) -> MolecularData:
        return super().__getitem__(idx)  # type: ignore


if __name__ == "__main__":
    dataset = SmilesTabular.read_csv(
        "data/mcl1/raw/mcl1_smiles.csv",
        name="mcl1",
        smiles_column_name="smiles",
        target_column_name="target",
    )
    print([dataset[i] for i in range(10)])
    print(len(dataset))
    print(dataset[0].z, dataset[0].pos)
