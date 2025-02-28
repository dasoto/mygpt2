from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import tiktoken
from dataloader import DataLoaderLite


@dataclass
class GPTConfig:
    block_size: int = 1024  # max_sequence_length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # Batch Size, sequence length, embedding dimensionality
        # Calculate query, key, values for all heads
        # nh is number of heads, hs is head size and C (number of channels) = nh * hs
        # Ex for GPT2 (124M) => nh = 12 *  hs= 64 => C = 768
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T,  hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # reassemble all head outputs side by side
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, "NANOGPT_SCALE_INIT"):
            std *= (2 * self.config.n_layer) ** -0.5
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        # Forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape T
        pos_emb = self.transformer.wpe(pos)  # pos embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # tok embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from higgingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print("Loading weights from pretrained gpt: %s" % model_type)
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # Always 50257 for GPT models checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard the mask buff

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()

        # Transformer GPT2 implementation use Conv1d instead of linear layers.
        # so those weights need to be transposed

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        print(f"sd {len(sd_keys)}")
        print(f"sdhf {len(sd_keys_hf)}")

        assert len(sd_keys) == len(sd_keys_hf)

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model


def run_inference(
    model, device: str, prompt: str, max_tokens: int = 30, num_sequences: int = 1
):
    # print(model)
    model.eval()
    model.to(device)

    # prefix tokens
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens.to(device)

    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    while x.size(1) < max_tokens:
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

    decoded = enc.decode_batch(x.tolist())
    for response in decoded:
        print(">", response)


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        print("GPU Found")
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple GPU Found")
        device = "mps"

    print(f"Using device: {device}")


if __name__ == "__main__":

    # model = GPT.from_pretrained("gpt2")
    device = get_device()
    # model = GPT(GPTConfig())
    # start_time = time.time()
    # run_inference(model, device, "Hello, how are you today?", 30, 5)
    # print(f"It took: {time.time() - start_time}")

    # Data loading
    B, T = 4, 1024

    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)
    elif device == "mps":
        torch.mps.manual_seed(1337)

    dataloader = DataLoaderLite("./dataset/input.txt", B, T)
    torch.set_float32_matmul_precision("high")

    model = GPT(GPTConfig())
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    start_training = time.time()
    for i in range(10):
        t0 = time.time()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        print(
            f"step {i}, loss: {loss.item()}, step_time={dt:.2f}ms, toks/sec={B * T / dt}"
        )

    print(f"Training time took: {time.time() - start_training} seconds")

    print("Starting Inference")
    run_inference(model, device, "Hello, how are you today?", 30, 5)
    print("End Inference")
