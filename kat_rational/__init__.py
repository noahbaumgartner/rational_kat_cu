from .kat_1dgroup_torch import KAT_Group_Torch

try:
    from .kat_1dgroup_triton import RationalTriton1DGroup, KAT_Group
    from .kat_2dgroup_triton import KAT_Group2D
except ImportError:
    class KAT_Group(KAT_Group_Torch):
        def __init__(self, num_groups=8, mode="gelu", device=None):
            super().__init__(num_groups=num_groups, mode=mode)
