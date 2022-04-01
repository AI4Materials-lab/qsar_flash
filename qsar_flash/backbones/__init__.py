from flash.core.registry import FlashRegistry

from qsar_flash.backbones.schnet import SchNetBackbone

MOLECULAR_GRAPH_BACKBONES = FlashRegistry("backbones")


@MOLECULAR_GRAPH_BACKBONES(name="SchNet")
def load_schnet(
    hidden_channels: int = 128,
    num_filters: int = 128,
    num_interactions: int = 6,
    num_gaussians: int = 50,
    cutoff: float = 10,
    max_num_neighbors: int = 32,
    **_
):
    schnet = SchNetBackbone(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        max_num_neighbors=max_num_neighbors,
    )

    return (
        schnet,
        hidden_channels,
    )
