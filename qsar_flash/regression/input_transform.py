from typing import Any, Callable, Dict, List

from torch.utils.data.dataloader import default_collate

from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.imports import _GRAPH_AVAILABLE
from flash.graph.classification.input_transform import PyGTransformAdapter

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Batch
    from torch_geometric.transforms import NormalizeFeatures
else:
    Data = object


class GraphRegressionInputTransform(InputTransform):
    @staticmethod
    def _pyg_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        inputs = Batch.from_data_list([sample[DataKeys.INPUT] for sample in samples])
        if DataKeys.TARGET in samples[0]:
            targets = default_collate([sample[DataKeys.TARGET] for sample in samples])
            return {DataKeys.INPUT: inputs, DataKeys.TARGET: targets}
        return {DataKeys.INPUT: inputs}

    def collate(self) -> Callable:
        return self._pyg_collate

    def per_sample_transform(self) -> Callable:
        return PyGTransformAdapter(NormalizeFeatures())
