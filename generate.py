import torch
from tokenizer import LumivexTokenizer
from model import LumivexModel

print("Loading LUMIVEX...")

tokenizer = LumivexTokenizer()
model = LumivexModel(vocab_size=100, context_length=32)

checkpoint = "lumivex_checkpoint.pt"
state = torch.load(checkpoint, map_location="cpu")

if isinstance(state, dict) and "model_state_dict" in state:
    model.load_state_dict(state["model_state_dict"])
elif isinstance(state, dict):
    model.load_state_dict(state)
else:
    model = state

model.eval()

text = "Hello LUMIVEX"
tokens = tokenizer.encode(text)
input_ids = torch.tensor([tokens], dtype=torch.long)

with torch.no_grad():
    output = model(input_ids)

print("LUMIVEX inference test successful.")
print("Prompt:", text)
print("Input tokens:", len(tokens))
print("Output shape:", output.shape)

next_token = torch.argmax(output[:, -1, :], dim=-1)
print("Next token ID:", next_token.item())

if hasattr(tokenizer, "decode"):
    print("Next token:", tokenizer.decode([next_token.item()]))
