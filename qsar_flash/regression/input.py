from typing import Any, Dict, Optional

from torch.utils.data import Dataset

from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.samples import to_sample
from flash.core.utilities.imports import _GRAPH_AVAILABLE, requires
from flash.core.data.utils import _STAGES_PREFIX

if _GRAPH_AVAILABLE:
    from torch_geometric.data import Data


def _get_num_features(sample: Dict[str, Any]) -> Optional[int]:
    """Get the number of features per node in the given dataset."""
    data = sample[DataKeys.INPUT]
    data = data[0] if isinstance(data, tuple) else data
    return getattr(data, "num_node_features", None)


class GraphRegressionDatasetInput(Input):
    @requires("graph")
    def load_data(self, dataset: Dataset) -> Dataset:
        if not self.predicting:
            self.num_features = _get_num_features(self.load_sample(dataset[0]))

        return dataset

    def load_sample(self, sample: Any) -> Dict[str, Any]:
        if isinstance(sample, Data):
            sample = (sample, sample.y)
        sample = to_sample(sample)
        return sample

    def _call_load_sample(self, sample: Any) -> Any:
        return getattr(self, f"{_STAGES_PREFIX[self.running_stage]}_load_sample")(sample)
