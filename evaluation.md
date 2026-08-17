# LUMIVEX Evaluation Benchmark

Every major LUMIVEX model version should be evaluated using:

1. Validation loss
2. Perplexity
3. Generation test
4. Tokenizer compatibility
5. Checkpoint loading

## Current benchmark

Model: LUMIVEX V3

The validation dataset must remain stable when comparing
model versions. If the benchmark changes, record the change
before comparing scores.

Lower validation loss and lower perplexity generally indicate
better performance on the validation corpus.
