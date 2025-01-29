from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, StaticCache
import torch


COMPRESSED_LENGTH = 16

def main():
    device = "cuda"
    #model_name = "NousResearch/Meta-Llama-3.1-8B"
    model_name = "allenai/OLMo-2-1124-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.bfloat16).to(device)
    model.forward = torch.compile(model.forward)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    cache= StaticCache(config=model.config, max_batch_size=1, max_cache_len=64000, device=device, dtype=model.dtype)

    thoughts = []
    for i in range(2, -1, -1):
        thoughts.append(torch.load(f"cache_{i}.pt"))

    while ((command := input("Input:\n")) != "/exit") and command:
        match command:
            case "/clear":
                for k, v in zip(cache.key_cache, cache.value_cache):
                    k.zero_()
                    v.zero_()
            case _:
                if command.startswith("/fill"):
                    num = int(command.split()[1])
                    pos = 0
                    for i in range(num):
                        for j, (k, v) in enumerate(zip(*thoughts[i])):
                            assert k.shape == v.shape
                            cache.key_cache[j][:, :, pos: pos + k.shape[2]].copy_(k)
                            cache.value_cache[j][:, :, pos: pos + v.shape[2]].copy_(v)
                        pos += thoughts[i][0][0].shape[2]
                    print("Filled up to position", pos)
                    continue

                pos = cache.get_seq_length()
                print(f"[Start generation from position {pos}]")
                inp = tokenizer(["\n" + command + "\n<|assistant|>\n"], return_tensors="pt", add_special_tokens=False).input_ids
                padded_inp = torch.zeros((1, pos + inp.shape[1]), dtype=torch.int32, device=device)
                padded_inp[:, -inp.shape[1]:].copy_(inp)
                res = model.generate(
                    input_ids=padded_inp,
                    use_cache=True, past_key_values=cache,
                    max_new_tokens=200
                )
                print(tokenizer.batch_decode(res[:, pos:])[0])


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
