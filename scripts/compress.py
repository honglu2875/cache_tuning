from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import torch
from contextlib import nullcontext
import torch.distributed._functional_collectives as fcol
from torch.optim import Adam
from torch.nn.functional import avg_pool1d
import torch
import tqdm
import math
import wandb
from transformers.cache_utils import DynamicCache, StaticCache
from cache_tuning.sample_prompts import QUICKSORT_PROMPT as PROMPT
from cache_tuning.data import TokenizedDataLoader


BATCH_SIZE = 16 
MICRO_BS = 4
COMPRESSED_LENGTH = 16
SEQ_LEN = 1024
MAX_SEQ_LEN = 2048
TRAIN_STEP = 1000


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

    #model_name = "NousResearch/Llama-3.2-1B"
    #model_name = "NousResearch/Meta-Llama-3.1-8B"
    model_name = "allenai/OLMo-2-1124-7B-Instruct"
    # I'd rather waste a cpu copy than using `accelerate`
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    NUM = 500
    past_key_values = StaticCache(config=model.config, max_batch_size=MICRO_BS, max_cache_len=MAX_SEQ_LEN + NUM * COMPRESSED_LENGTH, device=device, dtype=model.dtype)

    prompt = tokenizer(open("out.txt", "r").read()).input_ids
    print("Finished tokenization.")

    for param in model.parameters():
        param.requires_grad = False


    ds = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample").shuffle(42)
    dl = TokenizedDataLoader(ds["train"], tokenizer, seq_len=1024)

    wandb.init(
        project="cache_tuning",
        config={
            "model": model_name,
            "num_caches": NUM,
            "batch_size": BATCH_SIZE,
            "micro_bs": MICRO_BS,
            "compressed_len": COMPRESSED_LENGTH,
            "seq_len": SEQ_LEN,
            "max_seq_len": MAX_SEQ_LEN,
            "train_step": TRAIN_STEP,
        },
        mode="disabled" if rank > 0 else "online",
    )
    FILL_LEN = MAX_SEQ_LEN - SEQ_LEN
    for idx in range(NUM):
        tok_input = torch.tensor([prompt[idx * SEQ_LEN: (idx + 1) * SEQ_LEN]], device=device)
        text_prompt = tokenizer.decode(prompt[idx * SEQ_LEN: (idx + 1) * SEQ_LEN])
        start = idx * COMPRESSED_LENGTH
        # populate cache
        repeated_tok_input = tok_input.repeat(MICRO_BS, 1)
        with torch.no_grad():
            output = model(
                repeated_tok_input,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=torch.arange(tok_input.shape[1], device=device) + start,
            )

        # init compressed cache
        def _fn(tensor):
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

        init_keys = [_fn(k[:1, :, : start + SEQ_LEN]) for k in past_key_values.key_cache]
        init_vals = [_fn(v[:1, :, : start + SEQ_LEN]) for v in past_key_values.value_cache]

        

        
        train_iter = batch(iter(dl))

        params = init_keys + init_vals
        for p in params:
            p.requires_grad = True
        opt = Adam(params, lr=0.05)


        #compiled = torch.compile(model)


        static_inputs = torch.zeros((MICRO_BS, FILL_LEN), dtype=torch.int64, device=device)

        torch.cuda.synchronize()
        torch.distributed.barrier(group=process_group, device_ids=[rank])

        line1 = tqdm.tqdm(bar_format='{desc}{postfix}', position=0) if rank == 0 else nullcontext()
        line2 = tqdm.tqdm(bar_format='{desc}{postfix}', position=1) if rank == 0 else nullcontext()
        pbar = tqdm.tqdm(total=TRAIN_STEP, position=2) if rank == 0 else nullcontext()
        losses = []
        grad_norms = []
        with line1, line2, pbar:
            opt.zero_grad(set_to_none=True)
            loss, grad_sq = None, None
            for i, sample in zip(range(TRAIN_STEP), train_iter):
                static_inputs.copy_(sample[0][:, :FILL_LEN])

                buff_keys, buff_vals = [], []
                for i in range(len(past_key_values.key_cache)):
                    k, v = past_key_values.key_cache[i], past_key_values.value_cache[i]
                    kk, vv = init_keys[i], init_vals[i]
                    buff_keys.append(torch.concat([k[:1, :, :start], kk], dim=2))
                    buff_vals.append(torch.concat([v[:1, :, :start], vv], dim=2))
                loss = train(model, static_inputs, past_key_values, buff_keys, buff_vals, start + SEQ_LEN, device)
                
                if rank == 0:
                    with torch.no_grad():
                        grad_sq = sum(p.grad.square().sum()/p.numel() for p in params) / len(params)
                        line1.set_description(f"loss: {loss.item()}, grad_norm: {math.sqrt(grad_sq)}.")
                        line2.set_description(f"slice of key cache: {[f'{k.item():.4f}' for k in init_keys[0][0, 0, :15, 0]]}.")
                        losses.append(loss.item())
                        grad_norms.append(math.sqrt(grad_sq.item()))
                        pbar.update(1)

                if (i + 1) % GRAD_ACC == 0:
                    # All reduce gradients
                    for p in params:
                        gradient = p.grad / process_group.size()
                        gradient = fcol.all_reduce(gradient, "sum", process_group)
                        p.grad.copy_(gradient)
                    opt.step()
                    opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            for i in range(len(past_key_values.key_cache)):
                k, v = past_key_values.key_cache[i], past_key_values.value_cache[i]
                kk, vv = init_keys[i], init_vals[i]
                k[:, :, start: start + COMPRESSED_LENGTH].copy_(kk)
                k[:, :, start + COMPRESSED_LENGTH :].zero_()
                v[:, :, start: start + COMPRESSED_LENGTH].copy_(vv)
                v[:, :, start + COMPRESSED_LENGTH :].zero_()

        torch.cuda.synchronize()
        torch.distributed.barrier(group=process_group, device_ids=[rank])

        if rank == 0:
            print("----- eval -----")
            next_start = start + COMPRESSED_LENGTH
            lemma = """Any finitely generated ring over a Noetherian ring is Noetherian. Any localization of a Noetherian ring is Noetherian."""
            eval_prompt = f"\nRead through the previous textbook and understand it well. Prove the following lemma: {lemma}\n<|assistant|>\n"
            eval_tok = tokenizer([eval_prompt] * MICRO_BS, return_tensors="pt", add_special_tokens=False).input_ids
            eval_inp = torch.zeros((MICRO_BS, next_start + eval_tok.shape[1]), dtype=eval_tok.dtype, device=device)
            eval_inp[:, -eval_tok.shape[1]:].copy_(eval_tok)
            
            #print("slice of key cache:", eval_cache.key_cache[0][0, 0, :, 0])

            torch.save((init_keys, init_vals), f"cache_{idx}.pt")
            
            res = model.generate(input_ids=eval_inp,
                                 use_cache=True, past_key_values=past_key_values,
                                 #cache_position=torch.arange(eval_tok.shape[1], device=device) + next_start,
                                 max_new_tokens=200)
            eval_gen = tokenizer.batch_decode(res[:, next_start:])[0]
            print(eval_gen)

            wandb.log({
                "avg_last_10_loss": sum(losses[-10:]) / 10,
                "avg_last_10_grad_norm": sum(grad_norms[-10:]) /10,
                "text_prompt": text_prompt,
                "eval_generation": eval_gen,
            })

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
