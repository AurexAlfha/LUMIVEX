import torch

from model import LumivexModel
from tokenizer import LumivexTokenizer


CHECKPOINT = "lumivex_instruction_v4.pt"

tokenizer = LumivexTokenizer()
model = LumivexModel()

state = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

model.load_state_dict(state)
model.eval()


def generate(prompt, max_new_tokens=40, temperature=0.8, top_k=8):
    prompt_tokens = tokenizer.encode(prompt)
    tokens = prompt_tokens.copy()

    for _ in range(max_new_tokens):

        context = tokens[-model.context_length:]

        input_ids = torch.tensor(
            [context],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:, -1, :]
        logits = logits / temperature

        # Repetition penalty
        for token_id in set(tokens[-20:]):
            logits[:, token_id] /= 1.15

        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=k)

        probabilities = torch.softmax(values, dim=-1)

        choice = torch.multinomial(
            probabilities,
            num_samples=1
        )

        next_token = indices.gather(
            -1,
            choice
        ).item()

        tokens.append(next_token)

    return tokenizer.decode(
        tokens[len(prompt_tokens):]
    )


print("LUMIVEX V4 GENERATION ENGINE READY")
print(generate("What is 2 plus 3?"))
print("GENERATION ENGINE TEST COMPLETE")
