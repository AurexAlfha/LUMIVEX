import torch

from tokenizer import LumivexTokenizer
from model import LumivexModel


tokenizer = LumivexTokenizer()
model = LumivexModel()

text = "Hello LUMIVEX"
tokens = tokenizer.encode(text)

input_ids = torch.tensor([tokens], dtype=torch.long)

output = model(input_ids)

print("LUMIVEX integration test successful.")
print("Text:", text)
print("Token count:", len(tokens))
print("Input shape:", input_ids.shape)
print("Output shape:", output.shape)
