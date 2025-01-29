import dataclasses
import torch
from typing import Any, Callable, Iterable
from transformers import PreTrainedModel, PretrainedConfig
from transformers.cache_utils import DynamicCache, StaticCache
from cache_tuning.memory import Memory
from cache_tuning.config import TrainingConfig


# Not planning to expose this as an argument for now... But can be done in the future.
SEQ_AXIS = 2


class CacheTrainer:
    def __init__(
        self,
        config: PretrainedConfig,
        training_config: TrainingConfig,
        device: Any = "cuda",
    ):
        torch.set_float32_matmul_precision('high')
        self.config = config
        self.training_config = training_config
        self.device = torch.device(device)
        self.trainable_memory: StaticCache | None = None
        self.optim = None

    def init_cache(self, seq_len: int | None = None):
        seq_len = seq_len or self.training_config.max_seq_len
        return StaticCache(self.config,
                           max_batch_size=self.training_config.batch_size,
                           max_cache_len=seq_len).to(self.device)

    def set_optimizer(self, optim: torch.optim.Optimizer):
        self.optim = optim

    def prefill(self, model: PreTrainedModel, input_ids: torch.Tensor, cache:
                StaticCache, start: int = 0):
        assert self.training_config.batch_size % input_ids.shape[0] == 0
        assert input_ids.shape[1] + start < self.training_config.max_seq_len
        assert cache.key_cache[0].shape[SEQ_AXIS]
        rep = self.training_config.batch_size // input_ids.shape[0]
        repeated_input = input_ids.repeat(rep, 1).to(self.device)

        with torch.inference_mode():
            output = model(
                repeated_input,
                use_cache=True,
                past_key_values=cache,
                cache_position=torch.arange(input_ids.shape[1],
                                            device=self.device) + start,
            )

        return output.past_key_values

    def train(self, data_iter: Iterable, target_prompt: torch.Tensor,
              num_steps: int):
        # Not putting assumptions and we demand the exact shape.
        assert target_prompt.shape == (1, self.training_config.max_seq_len)
        for i in range(num_steps):
            ...


    def set_up_trainable_cache(self) -> None:
        # NOTE: Keeping the initial values random is not optimal for cache tuning!
        # Please modify the initial values afterwards.
        if self.trainable_memory is None:
            raise ValueError("Please prefill before setting up trainable cache.")
        kv_n_head = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        nh, hd = kv_n_head, self.config.hidden_size // kv_n_head
        num_layers = len(self.trainable_memory.key_cache)
        # hardcode float32 for now because mem efficiency is not the biggest concern.
        match SEQ_AXIS:
            case 1:
                shape = (1, self.training_config.num_mem_tokens, nh, hd)
            case 2:
                shape = (1, nh, self.training_config.num_mem_tokens, hd)
            case _:
                raise
        keys = [torch.empty(shape, device=self.device, dtype=torch.float32) for _ in range(num_layers)]
        values = [torch.empty(shape, device=self.device, dtype=torch.float32) for _ in range(num_layers)]
        for t in keys + values:
            t.requires_grad = True
        self.trainable_memory = Memory(keys=keys, values=values)

