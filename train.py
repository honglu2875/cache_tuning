from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
from contextlib import nullcontext
import torch.distributed._functional_collectives as fcol
from torch.optim import Adam
from torch.nn.functional import avg_pool1d
import torch
import tqdm
from transformers.cache_utils import DynamicCache, StaticCache
from prompts import PROMPT
from data import DL


BATCH_SIZE = 16 
MICRO_BS = 16
COMPRESSED_LENGTH = 10
SEQ_LEN = 512
TRAIN_STEP = 100


GRAD_ACC = BATCH_SIZE // MICRO_BS

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
    # Massive speed boost from enabling fp32 tensor core
    torch.set_float32_matmul_precision('high')

    torch.distributed.init_process_group(backend="nccl")
    mesh_dp = torch.distributed.init_device_mesh("cuda", mesh_shape=(torch.distributed.get_world_size(),))
    process_group = mesh_dp.get_group()
    rank = torch.distributed.get_rank(process_group)
    device = torch.device(rank % 8)

    model_name = "NousResearch/Llama-3.2-1B"
    # I'd rather waste a cpu copy than using `accelerate`
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False


    ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample").shuffle(42)
    dl = DL(ds["train"], tokenizer)



    past_key_values = StaticCache(config=model.config, max_batch_size=MICRO_BS, max_cache_len=512, device=device, dtype=model.dtype)


    tok_input = tokenizer([PROMPT], return_tensors="pt", add_special_tokens=False).to(device)
    PROMPT_LEN = tok_input.input_ids.shape[1]
    FILL_LEN = SEQ_LEN - PROMPT_LEN
    # populate cache
    repeated_tok_input = tok_input.input_ids.repeat(MICRO_BS, 1)
    with torch.no_grad():
        output = model(repeated_tok_input, use_cache=True, past_key_values=past_key_values)

    # init compressed cache
    def _fn(shape, tensor):
        #return torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True)
        #return torch.zeros(shape, dtype=torch.float32, device=device, requires_grad=True)

        #stride = PROMPT_LEN // COMPRESSED_LENGTH
        #kernel_size = stride * 2
        #p = tensor.permute((0, 1, 3, 2))
        #p = p.reshape(tensor.shape[0], -1, tensor.shape[2])
        #contracted = avg_pool1d(p, kernel_size=kernel_size, stride=stride, ceil_mode=True).permute((0, 2, 1))
        #contracted = contracted[:, -COMPRESSED_LENGTH:].reshape(shape)
        #return contracted

        return tensor[:, :, -COMPRESSED_LENGTH:].clone()

    _, nh, _, hd = past_key_values.key_cache[0].shape
    init_keys = [_fn((1, nh, COMPRESSED_LENGTH, hd), k[:1, :, :PROMPT_LEN]) for k in past_key_values.key_cache]
    init_vals = [_fn((1, nh, COMPRESSED_LENGTH, hd), v[:1, :, :PROMPT_LEN]) for v in past_key_values.value_cache]


    
    train_iter = batch(iter(dl))

    params = init_keys + init_vals
    for p in params:
        p.requires_grad = True
    opt = Adam(params, lr=1)


    #compiled = torch.compile(model)


    static_inputs = torch.zeros((MICRO_BS, FILL_LEN), dtype=torch.int64, device=device)

    torch.cuda.synchronize()
    torch.distributed.barrier(group=process_group, device_ids=[rank])

    line1 = tqdm.tqdm(bar_format='{desc}{postfix}', position=0) if rank == 0 else nullcontext()
    line2 = tqdm.tqdm(bar_format='{desc}{postfix}', position=1) if rank == 0 else nullcontext()
    pbar = tqdm.tqdm(total=TRAIN_STEP, position=2) if rank == 0 else nullcontext()
    with line1, line2, pbar:
        opt.zero_grad(set_to_none=True)
        loss, grad_sq = None, None
        for i, sample in zip(range(TRAIN_STEP), train_iter):
            static_inputs.copy_(sample[0][:, :FILL_LEN])

            loss = train(model, static_inputs, past_key_values, init_keys, init_vals, PROMPT_LEN, device)
            
            if rank == 0:
                with torch.no_grad():
                    grad_sq = sum(p.grad.square().sum()/p.numel() for p in params) / len(params)
                    line1.set_description(f"loss: {loss.item()}, grad_norm: {torch.sqrt(grad_sq).item()}.")
                    line2.set_description(f"slice of key cache: {[f'{k.item():.4f}' for k in init_keys[0][0, 0, :20, 0]]}.")
                    pbar.update(1)

            if (i + 1) % GRAD_ACC == 0:
                # All reduce gradients
                for p in params:
                    gradient = p.grad / process_group.size()
                    gradient = fcol.all_reduce(gradient, "sum", process_group)
                    p.grad.copy_(gradient)
                opt.step()
                opt.zero_grad(set_to_none=True)


    torch.cuda.synchronize()
    torch.distributed.barrier(group=process_group, device_ids=[rank])

    if rank == 0:
        print("----- eval -----")

        eval_prompt = "\nRepeat the above code:"
        eval_tok = tokenizer([eval_prompt], return_tensors="pt", add_special_tokens=False).input_ids
        eval_inp = torch.zeros((1, COMPRESSED_LENGTH + eval_tok.shape[1]), dtype=eval_tok.dtype, device=device)
        eval_inp[:, -eval_tok.shape[1]:].copy_(eval_tok)
        
        eval_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=SEQ_LEN, device=device, dtype=model.dtype)

        with torch.no_grad():
            for i, (k, v) in enumerate(zip(init_keys, init_vals)):
                eval_cache.key_cache[i][:, :, :k.shape[2]].copy_(k)
                eval_cache.value_cache[i][:, :, :v.shape[2]].copy_(v)

            print("slice of key cache:", eval_cache.key_cache[0][0, 0, :, 0])
            print("infered seq len:", eval_cache.get_seq_length())

            torch.save((eval_cache.key_cache, eval_cache.value_cache), "cache.pt")
            
            res = model.generate(input_ids=eval_inp,
                                 use_cache=True, past_key_values=eval_cache,
                                 max_new_tokens=200)
            print(tokenizer.batch_decode(res[:, COMPRESSED_LENGTH:])[0])

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
