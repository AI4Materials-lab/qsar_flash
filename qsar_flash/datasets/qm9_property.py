from torch_geometric.datasets import QM9


class QM9Property(QM9):
    def __init__(self, property_index: int = None, **qm9_kwargs):
        super().__init__(**qm9_kwargs)
        if property_index is not None:
            self.data.y = self.data.y[:, property_index].clone()
