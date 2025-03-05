from absl import app
from absl import flags
import logging
from absl import logging as absl_logging
import time

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

    model = GPT(GPTConfig())
    print(f"Sending model to: {device}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = None
    if device == "cuda" and FLAGS.autocast and FLAGS.autocast_precision == "float16":
        scaler = torch.amp.GradScaler()

    start_training = time.time()
    for i in range(FLAGS.steps):
        t0 = time.time()
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if FLAGS.autocast:
            with torch.autocast(
                device_type=device, dtype=Precision[FLAGS.autocast_precision].value
            ):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000  # milliseconds
        print(
            f"step {i}, loss: {loss.item()}, step_time={dt:.2f}ms, toks/sec={(B * T / dt) * 1000:.2f}, device: {device}"
        )

    print(f"Training time took: {time.time() - start_training} seconds")
    if FLAGS.run_inference:
        print("Starting Inference")
        run_inference(model, device, "Hello, how are you today?", 30, 5)
        print("End Inference")


if __name__ == "__main__":
    app.run(main)
