from torch.utils.data import DataLoader
import torch

SEQ_LEN = 512
class DL(DataLoader):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self._all_tokens = []
        self._sample_idx = 0
        self.infinite = True

    def __iter__(self):
        max_buffer_token_len = 1 + SEQ_LEN

        while True:
            for sample in iter(self.dataset):
                # Use the dataset-specific text processor
                sample_tokens = self.tokenizer.encode(sample["text"])
                self._all_tokens.extend(sample_tokens)
                self._sample_idx += 1

                while len(self._all_tokens) >= max_buffer_token_len:
                    x = torch.LongTensor(self._all_tokens[:max_buffer_token_len])
                    # update tokens to the remaining tokens
                    self._all_tokens = self._all_tokens[max_buffer_token_len:]
                    input = x[:-1]
                    label = x[1:]
                    yield input, label

            if not self.infinite:
                print(f"Dataset has run out of data")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                print(f"Dataset is being re-looped")

