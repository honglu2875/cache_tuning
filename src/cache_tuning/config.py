import dataclasses
import tomllib
from pathlib import Path


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingConfig:
    batch_size: int
    # the maximal sequence length for the model
    max_seq_len: int
    # the length of every explicit text chunk
    text_len: int
    # number of "memory tokens", i.e., size of the compressed memory cache
    num_mem_tokens: int
    # Note: learning rates need to be fairly large
    lr: float
    # number of gradient steps
    steps: int

    @classmethod
    def from_toml(cls, path: Path | str) -> "TrainingConfig":
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls(**data)
