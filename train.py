from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
import torch.distributed._functional_collectives as fcol
from torch.optim import Adam
import torch
import tqdm
from transformers.cache_utils import DynamicCache, StaticCache
from prompts import PROMPT
from data import DL


BATCH_SIZE = 64
MICRO_BS = 16
COMPRESSED_LENGTH = 124
SEQ_LEN = 512

def batch(it):
    buffer = []
    for sample in it:
        buffer.append(sample)
        if len(buffer) >= MICRO_BS:
            yield (torch.stack([x[0] for x in buffer], dim=0), torch.stack([x[1] for x in buffer], dim=0))
            buffer.clear()

def _reset(init_keys, init_vals):
        cache = DynamicCache()
        for idx, (k, v) in enumerate(zip(init_keys, init_vals)):
            cache.update(k.repeat(MICRO_BS, 1, 1, 1), v.repeat(MICRO_BS, 1, 1, 1), idx)
        return cache

def train(model, inputs, ref_cache, init_keys, init_vals, pos_offset, device):
        cache = _reset(init_keys, init_vals)
        
        with torch.no_grad():
            ref_output = model(inputs, use_cache=True, past_key_values=ref_cache,
                               cache_position=torch.arange(inputs.shape[1], device=device) + pos_offset)
        output = model(inputs, use_cache=True, past_key_values=cache)
        loss = (output.logits - ref_output.logits).square().mean()
        loss.backward()
        return loss

def main():
    torch.distributed.init_process_group(backend="nccl")
    mesh_dp = torch.distributed.init_device_mesh("cuda", mesh_shape=(8,))
    process_group = mesh_dp.get_group()
    device = torch.device(torch.distributed.get_rank())

    model_name = "NousResearch/Llama-3.2-1B"
    # I'd rather waste a cpu copy than using `accelerate`
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False


    ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample").shuffle(42)
    dl = DL(ds["train"], tokenizer)



    past_key_values = StaticCache(config=model.config, max_batch_size=MICRO_BS, max_cache_len=512, device=device, dtype=model.dtype)


    tok_input = tokenizer([PROMPT], return_tensors="pt").to(device)
    # populate cache
    repeated_tok_input = tok_input.input_ids.repeat(MICRO_BS, 1)
    with torch.no_grad():
        output = model(repeated_tok_input, use_cache=True, past_key_values=past_key_values)

    # init compressed cache
    def _fn(*shape):
        #return torch.randn(shape, dtype=torch.float32, device="cuda", requires_grad=True)
        return torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=True)

    _, nh, _, hd = past_key_values.key_cache[0].shape
    init_keys = [_fn(1, nh, COMPRESSED_LENGTH, hd) for _ in range(len(past_key_values.key_cache))]
    init_vals = [_fn(1, nh, COMPRESSED_LENGTH, hd) for _ in range(len(past_key_values.value_cache))]


    
    train_iter = batch(iter(dl))

    params = init_keys + init_vals
    for p in params:
        p.requires_grad = True
    opt = Adam(params, lr=0.01)
    PROMPT_LEN = tok_input.input_ids.shape[1]
    FILL_LEN = SEQ_LEN - PROMPT_LEN
    GRAD_ACC = BATCH_SIZE // MICRO_BS


    compiled = torch.compile(model)


    static_inputs = torch.zeros((MICRO_BS, FILL_LEN), dtype=torch.int64, device=device)

    pbar = tqdm.tqdm(zip(range(100), train_iter))
    opt.zero_grad(set_to_none=True)
    for i, sample in pbar:
        static_inputs.copy_(sample[0][:, :FILL_LEN])

        loss = train(model, static_inputs, past_key_values, init_keys, init_vals, PROMPT_LEN, device)
        
        #if torch.distributed.get_rank() == 0:
        with torch.no_grad():
            grad_sq = sum(p.grad.square().sum()/p.numel() for p in params) / len(params)
        pbar.set_description(f"loss: {loss.item()}, grad_norm: {torch.sqrt(grad_sq).item()}")

        if (i + 1) % GRAD_ACC == 0:
            # All reduce gradients
            for p in params:
                gradient = p.grad / process_group.size()
                gradient = fcol.all_reduce(gradient, "sum", process_group)
                p.grad.copy_(gradient)
            opt.step()
            opt.zero_grad(set_to_none=True)


    print("----- eval -----")

    eval_prompt = "\nThe above code can be explained as follows:"
    eval_inp = tokenizer([PROMPT + eval_prompt], return_tensors="pt").input_ids.to(device)
    eval_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=512, device=device, dtype=model.dtype)

    with torch.no_grad():
        for i, (k, v) in enumerate(zip(init_keys, init_vals)):
            eval_cache.key_cache[i][:, :, :k.shape[2]].copy_(k)
            eval_cache.value_cache[i][:, :, :v.shape[2]].copy_(v)
        
        res = model.generate(input_ids=eval_inp,
                             use_cache=True, past_key_values=eval_cache,
                             max_new_tokens=200)
        print(tokenizer.batch_decode(res)[0])

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
