import math
import torch

block_size = 3
batch_size = 64
n_embd = 10
n_hidden = 200

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read()
vocab = list(set(text))
vocab.sort()
vocab_size = len(vocab)

map_char = {j: i for i, j in enumerate(vocab)}

stoi = lambda c: map_char[c]
itos = lambda i: vocab[i]
encode = lambda s: [stoi(c) for c in s]
decode = lambda i: ''.join([itos(j) for j in i])

data = torch.tensor(encode(text), dtype=torch.long)
stop = math.floor(len(text) * 0.9)
train = torch.tensor(encode(text[:stop]), dtype=torch.long)
val = torch.tensor(encode(text[stop:]), dtype=torch.long)

def get_batch(split):
    if split == "train":
        values = train
    else:
        values = val
    x_batchs = []
    y_batchs = []
    starts = torch.randint(high=len(values) - block_size, size=(batch_size,))
    for start in starts:
        x_batchs.append(values[start:start + block_size])
        y_batchs.append(values[start + block_size])
    x = torch.stack(x_batchs)
    y = torch.stack(y_batchs)
    return x, y


class Bigram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, n_embd)
        self.hidden = torch.nn.Linear(block_size * n_embd, n_hidden)
        self.out = torch.nn.Linear(n_hidden, vocab_size)

    def forward(self, v):
        embedded = self.emb(v).view(-1, block_size * n_embd)
        hidden = self.hidden(embedded)
        relu = torch.relu(hidden)
        return self.out(relu)


def generate(_model):
    with torch.no_grad():
        ids = data[:block_size].tolist()
        for i in range(0, 300):
            batch = torch.tensor([ids[-block_size:]])
            logits = _model(batch)
            prob = torch.softmax(logits, dim=-1)
            new_id = torch.multinomial(prob, 1)
            ids.append(new_id.item())
        print(decode(ids))


def get_loss(_model: Bigram, split: str) -> torch.Tensor:
    entrees, cibles = get_batch(split)
    logits = _model.forward(entrees)
    return torch.nn.functional.cross_entropy(logits, cibles)

model = Bigram()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
generate(model)
for i in range(0, 10000):
    loss = get_loss(model, "train")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if i % 1000 == 0:
    #     with torch.no_grad():
    #         loss_train = []
    #         loss_val = []
    #         for _ in range(0, 50):
    #             loss_train.append(get_loss(model, "train"))
    #             loss_val.append(get_loss(model, "val"))
    #         print(f"Round {i}:\nTrain: {torch.stack(loss_train).mean()}\nVal: {torch.stack(loss_val).mean()}")
generate(model)