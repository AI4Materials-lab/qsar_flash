from pathlib import Path
from typing import cast, Dict, Optional, Protocol, Tuple, Union

import ase.io as aio
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class MolecularData(Protocol):
    z: torch.Tensor
    pos: torch.Tensor
    y: torch.Tensor
    __num_nodes__: int


class FolderDataset(InMemoryDataset):
    def __init__(
        self,
        folderpath: Union[str, Path],
        target_df: pd.DataFrame,
        *,
        name: str,
        filetype: str = "xyz",
        root: str = "data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ) -> None:
        self.folderpath = Path(folderpath)
        self.target_df = target_df
        self.filetype = filetype
        self.original_root = Path(root)
        self.folder = self.original_root / name

        super().__init__(str(self.folder), transform, pre_transform, pre_filter)

        self.data, self.slices = self.load_processed_data()
        self.data.z = self.data.z.long()

    @classmethod
    def read_csv(
        cls,
        folderpath: Union[str, Path],
        reader: Union[str, Path],
        *,
        name: str,
        filetype: str = "xyz",
        filename_column_name: str = "filename",
        target_column_name: str = "target",
        root: str = "data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        **pandas_kwargs,
    ):

        df = cast(
            pd.DataFrame, pd.read_csv(reader, usecols=(filename_column_name, target_column_name), **pandas_kwargs)
        )
        df.set_index(filename_column_name, inplace=True)
        return cls(
            folderpath=folderpath,
            target_df=df,
            name=name,
            filetype=filetype,
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    @classmethod
    def read_df(
        cls,
        folderpath: Union[str, Path],
        df: pd.DataFrame,
        *,
        name: str,
        filetype: str = "xyz",
        filename_column_name: str = "filename",
        target_column_name: str = "target",
        root: str = "data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        df = df.loc[:, [filename_column_name, target_column_name]]
        df.set_index(filename_column_name, inplace=True)
        return cls(
            folderpath=folderpath,
            target_df=df,
            name=name,
            filetype=filetype,
            root=root,
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
        data_list = []
        for filename in tqdm(self.folderpath.iterdir(), desc="Load files", total=len(self.target_df)):
            filename = cast(Path, filename)
            if not filename.suffix.endswith(self.filetype):
                continue
            atoms = cast(Atoms, aio.read(filename, index=0))

            data: MolecularData = cast(MolecularData, Data())
            data.__num_nodes__ = len(atoms)
            data.z = torch.from_numpy(atoms.get_atomic_numbers()).to(torch.uint8)
            data.pos = torch.tensor(atoms.get_positions()).to(torch.float32)
            data.y = torch.tensor(self.target_df.loc[filename.name]).to(torch.float32)

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
    dataset = FolderDataset.read_csv(
        "/mnt/storage/new_materials/data/tmqm/xyz/xyz_molecules/",
        "id_prop.csv",
        name="tmqm",
        filename_column_name="filename",
        target_column_name="target",
    )
    print([dataset[i] for i in range(10)])
    print(len(dataset))
    print(dataset[0].y)
    print(dataset[0].z, dataset[0].pos)
