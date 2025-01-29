import dataclasses
import torch


@dataclasses.dataclass(frozen=True, slots=True)
class Memory:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    def __post_init__(self):
        assert self.keys[0].shape[0] == 1
        assert self.values[0].shape[0] == 1
        assert all(self.keys[0].shape == k for k in self.keys[1:])
        assert all(self.values[0].shape == v for v in self.values[1:])

    @property
    def length(self):
        return self.keys[0].shape[2]


