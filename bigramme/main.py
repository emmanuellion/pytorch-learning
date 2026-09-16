import math
import torch

# block_size = 8
# batch_size = 32

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

# data = torch.tensor(encode(text), dtype=torch.long)
# stop = math.floor(len(text) * 0.9)
# train = torch.tensor(encode(text[:stop]), dtype=torch.long)
# val = torch.tensor(encode(text[stop:]), dtype=torch.long)

# def get_batch(split):
#     if split == "train":
#         values = train
#     else:
#         values = val
#     x_batchs = []
#     y_batchs = []
#     starts = torch.randint(high=len(values) - block_size, size=(batch_size,))
#     for start in starts:
#         picked = values[start:start + block_size]
#         x_batchs.append(picked)
#         y_batchs.append(values[start+1:start + block_size + 1])
#     x = torch.stack(x_batchs)
#     y = torch.stack(y_batchs)
#     return x, y
#
#
# class Bigram(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.embedding = torch.nn.Embedding(vocab_size, vocab_size)
#
#     def forward(self, v):
#         return self.embedding(v)


count = torch.load('count.pt', weights_only=True)

# count = torch.zeros((vocab_size, vocab_size))
# for i, c in enumerate(text):
#     if i+1 < len(text):
#         c1 = c
#         c2 = text[i + 1]
#         count[stoi(c1), stoi(c2)] += 1
# torch.save(count, 'count.pt')

# soft = torch.softmax(count, dim=-1)
count += 1
prob = count / count.sum(dim=1, keepdim=True)
top = torch.topk(prob[stoi("q")], k=5, dim=-1)
idx = top.indices
vals = top.values
# for i in range(0, len(top.indices)):
#     print(f"'{itos(idx[i].item())}': {vals[i].item()}")

ids = [stoi("L")]
for i in range(0, 300):
    last_prob = prob[ids[-1]]
    new_id = torch.multinomial(last_prob, 1)
    ids.append(new_id.item())
print(''.join(decode(ids)))
x = ids[:-1]
y = ids[1:]
print(prob[x, y].shape)