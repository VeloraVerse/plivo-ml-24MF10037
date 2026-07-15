Our best configuration consists of a 4-layer GPT model with 6 attention heads, embedding size 192, and a context window of 256.
It uses a custom Byte-level BPE tokenizer (vocab 1024) trained on the corpus, which compresses Hindi and English text by 1.91x compared to the baseline's byte-level tokenizer.
This BPE tokenizer effectively doubles the context window and increases token training throughput.
Tying embedding and LM head weights saves 196,608 parameters, enabling a deeper 4-layer network within the 2,000,000 parameter budget.
We replaced absolute positional embeddings with Rotary Position Embeddings (RoPE), which improves generalization and saves an additional 49,152 parameters.
We swapped LayerNorm with RMSNorm to reduce parameter overhead and speed up execution.
We replaced the standard GELU MLP with a SwiGLU MLP, increasing the model's representation capacity for the same parameter count.
The training uses AdamW (weight decay 0.1) and a Cosine learning rate scheduler with a peak LR of 2e-3 and 200 steps of warmup.
An increased batch size of 16 stabilizes gradients and accelerates convergence.
Together, these upgrades enable rapid convergence, achieving a dev set bits-per-byte (bpb) of ~1.96 in 500 steps, dramatically outperforming the baseline.
