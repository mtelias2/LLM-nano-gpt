# creating LLm nano GPT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from sklearn.model_selection import train_test_split
import os
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
file_name = "input.txt"


file_path = os.path.join(os.getcwd(), "data", file_name)


if not os.path.exists(file_path):
    # Download the file from https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    urllib.request.urlretrieve(url, file_path)

else:
    # File exists, continue with the rest of the code
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)

stoi = {c: i for i, c in enumerate(unique_chars)}
itos = {i: c for i, c in enumerate(unique_chars)}

encoder = lambda s: [stoi[x] for x in s]
decoder = lambda l: "".join([itos[i] for i in l])

data = encoder(text)

torch_data = torch.tensor(data)

train_data, test_data = train_test_split(torch_data, test_size=0.1, shuffle=False)
# set seed
torch.manual_seed(1337)


block_size = 8  # the charachter block size for the transformer
batch_size = 4  # the batch size


class BigramModel(nn.Module):
    """
    A class representing a Bigram language model.
    """

    def __init__(self, vocab_size):
        """
        Initialize the BigramModel with the given vocabulary size.

        Args:
        - vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass of the BigramModel.

        Args:
        - idx (torch.Tensor): The input indices representing the current context.
        - targets (torch.Tensor): The target indices.

        Returns:
        - logits (torch.Tensor): The output logits.
        - loss (torch.Tensor): The loss (if targets is not None).
        """
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            targets = targets.view(-1)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens based on the input indices.

        Args:
        - idx (torch.Tensor): The input indices representing the current context.
        - max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
        - idx (torch.Tensor): The generated indices.
        """
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(split):
    """
    Get a batch of data from the specified split.

    Args:
    - split (str): The split to get the data from.

    Returns:
    - x (torch.Tensor): The input data.
    - y (torch.Tensor): The target data.
    """
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    """
    Estimate the loss of the model. and print every 100 steps

    Returns:
    - out (dict): A dictionary containing the losses for different splits.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = BigramModel(vocab_size)
optimizer = adam.Adam(model.parameters(), lr=1e-3)

batch_size = 32
eval_iters = 200
for steps in range(10000):
    if steps % eval_iters == 0:
        losses = estimate_loss()
        print(steps, losses["train"], losses["val"])

    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print loss

context = torch.zeros((1, 1), dtype=torch.long)
print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))
