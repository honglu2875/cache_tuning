from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
from contextlib import nullcontext
import torch.distributed._functional_collectives as fcol
from torch.optim import Adam
import torch
import tqdm
import math
import random
from transformers.cache_utils import DynamicCache, StaticCache
from cache_tuning.sample_prompts import QUICKSORT_PROMPT as PROMPT
from cache_tuning.data import TokenizedDataLoader


BATCH_SIZE =  16
MICRO_BS = 4
COMPRESSED_LENGTH = 32
SEQ_LEN = 512
MAX_SEQ_LEN = 1024
TRAIN_STEP = 500


GRAD_ACC = BATCH_SIZE // MICRO_BS
PROMPT = """# Preface
The  author  offers  this  brief  Life  of  Martin 
Luther  as  her  contribution  to  the  literature  of 
the  Four  Hundredth  Anniversary  of  the  Refor- 
mation. The  volume  contains  no  original  ma- 
terial, but  is  intended  to  serve  as  an  introduc- 
tion to  the  longer,  richer,  and  more  scholarly 
records  of  a  great  life  which  abound  and  to 
the  noble  writings  of  the  Reformer  himself. 
Grateful  acknowledgment  is  made  to  the  biog- 
raphers of  Luther,  especially  to  Dr.  Henry  E. 
Jacobs,  to  Dr.  Preserved  Smith,  and  to  Heinrich 
Bohmer,  the  author  of  Luther  in  the  Light  of 
Recent  Research."""

def batch(it):
    prefetch_interval = 1024
    buffer = []
    for sample in it:
        buffer.append(sample)
        if len(buffer) >= prefetch_interval:
            random.shuffle(buffer)
            for i in range(0, len(buffer), MICRO_BS):
                out = buffer[i: i + MICRO_BS]
                if len(out) != MICRO_BS:
                    break
                yield (torch.stack([x[0] for x in out], dim=0), torch.stack([x[1] for x in out], dim=0))
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

    #model_name = "NousResearch/Llama-3.2-1B"
    model_name = "NousResearch/Meta-Llama-3.1-8B"
    # I'd rather waste a cpu copy than using `accelerate`
    model = AutoModelForCausalLM.from_pretrained(model_name).to(torch.bfloat16).to(device)
    #compiled = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for param in model.parameters():
        param.requires_grad = False


    ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample").shuffle(42)
    dl = TokenizedDataLoader(ds["train"], tokenizer, seq_len=MAX_SEQ_LEN)



    past_key_values = StaticCache(config=model.config, max_batch_size=MICRO_BS, max_cache_len=MAX_SEQ_LEN, device=device, dtype=model.dtype)

    tok_input = tokenizer([PROMPT], return_tensors="pt", add_special_tokens=False).to(device)
    # populate cache
    repeated_tok_input = tok_input.input_ids.repeat(MICRO_BS, 1)
    with torch.no_grad():
        output = model(repeated_tok_input, use_cache=True, past_key_values=past_key_values)


    for epoch in range(10000):

        if epoch > 0:
            prompt_len = SEQ_LEN
        else:
            prompt_len = tok_input.input_ids.shape[1]
        fill_len = MAX_SEQ_LEN - prompt_len

        # init compressed cache
        def _fn(shape, tensor):
            return tensor[:, :, -COMPRESSED_LENGTH:].clone()

        _, nh, _, hd = past_key_values.key_cache[0].shape
        init_keys = [_fn((1, nh, COMPRESSED_LENGTH, hd), k[:1, :, :prompt_len]) for k in past_key_values.key_cache]
        init_vals = [_fn((1, nh, COMPRESSED_LENGTH, hd), v[:1, :, :prompt_len]) for v in past_key_values.value_cache]


        
        train_iter = batch(iter(dl))

        params = init_keys + init_vals
        for p in params:
            p.requires_grad = True
        opt = Adam(params, lr=0.02)

        static_inputs = torch.zeros((MICRO_BS, fill_len), dtype=torch.int64, device=device)

        torch.cuda.synchronize()
        torch.distributed.barrier(group=process_group, device_ids=[rank])

        line1 = tqdm.tqdm(bar_format='{desc}{postfix}', position=0) if rank == 0 else nullcontext()
        line2 = tqdm.tqdm(bar_format='{desc}{postfix}', position=1) if rank == 0 else nullcontext()
        pbar = tqdm.tqdm(total=TRAIN_STEP, position=2) if rank == 0 else nullcontext()
        with line1, line2, pbar:
            opt.zero_grad(set_to_none=True)
            loss, grad_sq = None, None
            for i, sample in zip(range(TRAIN_STEP), train_iter):
                static_inputs.copy_(sample[0][:, :fill_len])

                loss = train(model, static_inputs, past_key_values, init_keys, init_vals, prompt_len, device)
                
                if rank == 0:
                    with torch.no_grad():
                        grad_sq = sum(p.grad.square().sum()/p.numel() for p in params) / len(params)
                        line1.set_description(f"loss: {loss.item()}, grad_norm: {math.sqrt(grad_sq)}.")
                        line2.set_description(f"slice of key cache: {', '.join(f'{k.item():.4f}' for k in init_keys[0][0, 0, :20, 0])}.")
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

        past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_SEQ_LEN, device=device, dtype=model.dtype)


        if rank == 0:
            #eval_prompt = "\n## Advanced topic\n\nGiven the previous knowledge, we can go one step forward:\n"
            eval_prompt = "\n"
            eval_tok = tokenizer([eval_prompt], return_tensors="pt", add_special_tokens=False).input_ids
            eval_inp = torch.zeros((1, COMPRESSED_LENGTH + eval_tok.shape[1]), dtype=eval_tok.dtype, device=device)
            eval_inp[:, -eval_tok.shape[1]:].copy_(eval_tok)
        

            with torch.no_grad():
                for i, (k, v) in enumerate(zip(init_keys, init_vals)):
                    past_key_values.key_cache[i][:, :, :k.shape[2]].copy_(k)
                    past_key_values.value_cache[i][:, :, :v.shape[2]].copy_(v)

                
                res = model.generate(
                    input_ids=eval_inp,
                    use_cache=True, past_key_values=past_key_values,
                    max_new_tokens=SEQ_LEN - eval_inp.shape[1],
                    min_new_tokens=SEQ_LEN - eval_inp.shape[1],
                    do_sample=True,
                    temperature=0.5,
                )

                print("----- generate -----")
                print(tokenizer.batch_decode(res[:, COMPRESSED_LENGTH:])[0])

                torch.save((init_keys, init_vals), f"cache_{epoch}.pt")
                with open(f"log_{rank}.txt", "a") as f:
                    f.write(f"\n--- epoch {epoch} ---\n")
                    f.write(tokenizer.batch_decode(res[:, eval_inp.shape[1]:])[0])

        torch.cuda.synchronize()
        torch.distributed.barrier(group=process_group, device_ids=[rank])

        past_key_values.key_cache = [fcol.all_reduce(k, "sum", process_group).repeat(MICRO_BS, 1, 1, 1) for k in past_key_values.key_cache]
        past_key_values.value_cache = [fcol.all_reduce(v, "sum", process_group).repeat(MICRO_BS, 1, 1, 1) for v in past_key_values.value_cache]

        torch.cuda.synchronize()
        torch.distributed.barrier(group=process_group, device_ids=[rank])


    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
