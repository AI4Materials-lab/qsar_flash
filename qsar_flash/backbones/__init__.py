from flash.core.registry import FlashRegistry

from qsar_flash.backbones.schnet import SchNetBackbone
from qsar_flash.backbones.dimenet import DimeNetBackbone
from qsar_flash.backbones.dimenet_plus_plus import DimeNetPlusPlusBackbone

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


@MOLECULAR_GRAPH_BACKBONES(name="DimeNet")
def load_dimenet(
    hidden_channels=128,
    out_channels=128,
    num_blocks=6,
    num_bilinear=8,
    num_spherical=7,
    num_radial=6,
    cutoff=10.0,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=2,
    num_output_layers=3,
    **_
):
    dimenet = DimeNetBackbone(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        num_bilinear=num_bilinear,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_output_layers=num_output_layers,
    )

    return (
        dimenet,
        out_channels,
    )


@MOLECULAR_GRAPH_BACKBONES(name="DimeNet++")
def load_dimenetpp(
    hidden_channels=128,
    out_channels=128,
    int_emb_size=64,
    out_emb_channels=256,
    num_blocks=6,
    basis_emb_size=8,
    num_spherical=7,
    num_radial=6,
    cutoff=10.0,
    envelope_exponent=5,
    num_before_skip=1,
    num_after_skip=2,
    num_output_layers=3,
    **_
):
    dimenetpp = DimeNetPlusPlusBackbone(
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_blocks=num_blocks,
        int_emb_size=int_emb_size,
        basis_emb_size=basis_emb_size,
        out_emb_channels=out_emb_channels,
        num_spherical=num_spherical,
        num_radial=num_radial,
        cutoff=cutoff,
        envelope_exponent=envelope_exponent,
        num_before_skip=num_before_skip,
        num_after_skip=num_after_skip,
        num_output_layers=num_output_layers,
    )

    return (
        dimenetpp,
        out_channels,
    )
