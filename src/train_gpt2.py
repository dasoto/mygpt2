from absl import app
from absl import flags
import logging
from absl import logging as absl_logging
import time
import math

import torch
from gpt2 import GPT, GPTConfig, run_inference
from dataloader import DataLoaderLite
import tiktoken

from enum import Enum


class Precision(Enum):
    bfloat16 = torch.bfloat16
    float16 = torch.float16


FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 1, "Batch Size")
flags.DEFINE_integer("seq_length", 1024, "Sequence Length")
flags.DEFINE_integer("steps", 10, "Number of steps to run")
flags.DEFINE_enum(
    "device",
    "auto",
    ["auto", "gpu", "cpu", "mps"],
    "Set the logging level.",
)
flags.DEFINE_enum(
    "log_level",
    "INFO",
    ["DEBUG", "INFO"],
    "Set the logging level.",
)
flags.DEFINE_enum(
    "precision",
    "highest",
    ["highest", "high", "medium"],
    "Set the logging level.",
)
flags.DEFINE_bool(
    "run_inference",
    False,
    "Run Inference at the end",
)
flags.DEFINE_bool(
    "autocast",
    False,
    "Enable autocast for training",
)
flags.DEFINE_enum(
    "autocast_precision",
    "bfloat16",
    ["bfloat16", "float16"],
    "Set the precision for autocast",
)
flags.DEFINE_bool(
    "compile",
    False,
    "Enable torch.compile for training",
)


def get_device():
    device = "cpu"
    if torch.cuda.is_available():
        print("GPU Found")
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple GPU Found")
        device = "mps"

    print(f"Using device: {device}")
    return device


def main(argv):
    # model = GPT.from_pretrained("gpt2")
    if FLAGS.device == "auto":
        device = get_device()
    else:
        device = FLAGS.device
    # model = GPT(GPTConfig())
    # start_time = time.time()
    # run_inference(model, device, "Hello, how are you today?", 30, 5)
    # print(f"It took: {time.time() - start_time}")

    # Data loading
    B, T = FLAGS.batch_size, FLAGS.seq_length

    torch.manual_seed(1337)
    if device == "cuda":
        torch.cuda.manual_seed(1337)
    elif device == "mps":
        torch.mps.manual_seed(1337)

    dataloader = DataLoaderLite("./dataset/input.txt", B, T)
    torch.set_float32_matmul_precision(FLAGS.precision)

    model = GPT(GPTConfig(vocab_size=50304))
    print(f"Sending model to: {device}")
    model.to(device)
    if FLAGS.compile:
        print("Compiling model")
        model = torch.compile(model)

    def get_lr(step: int) -> float:
        # get the learning rate using cosine decay
        # 1. linear warmup for some steps
        # 2. then cosine decay to min_steps that is 10% of max_lr
        max_steps = 0.9 * FLAGS.steps
        warmup_steps = max_steps // 10
        max_lr = 6e-4
        min_lr = 0.1 * max_lr
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        if step > max_steps:
            return min_lr
        else:
            decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (max_lr - min_lr)

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    # )
    optimizer = model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device=device
    )

    scaler = None
    if device == "cuda" and FLAGS.autocast and FLAGS.autocast_precision == "float16":
        scaler = torch.amp.GradScaler()

    start_training = time.time()
    for step in range(FLAGS.steps):
        t0 = time.time()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if FLAGS.autocast:
            with torch.autocast(
                device_type=device, dtype=Precision[FLAGS.autocast_precision].value
            ):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        if scaler:
            scaler.scale(loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        print(
            f"step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} |norm: {norm:.4f} | step_time={dt:.2f}ms, toks/sec={(B * T / dt) * 1000:.2f}, device: {device}"
        )

    print(f"Training time took: {time.time() - start_training} seconds")
    if FLAGS.run_inference:
        print("Starting Inference")
        run_inference(model, device, "Hello, how are you today?", 30, 5)
        print("End Inference")


if __name__ == "__main__":
    app.run(main)
