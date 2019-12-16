import torch
from torch.nn.parameter import Parameter


embedding = torch.nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5],[1,2,4,5],[4,3,2,9]])
print(embedding.weight.detach())
print(input)
a = embedding(input)
print(a)

