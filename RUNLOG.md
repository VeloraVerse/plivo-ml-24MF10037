# LLM Training Run Log

## Run 0: Baseline
- **Hypothesis**: The mediocre baseline GPT model using byte-level tokenizer and constant learning rate will serve as our starting benchmark.
- **What Changed**: None (Baseline configuration).
- **Steps**: 2000
- **Parameters**: 1,339,840
- **Dev bpb**: 2.3718
- **Conclusion**: The baseline is highly inefficient: it represents Hindi characters with 3 tokens (bytes), wastes context window, has no LR schedule/warmup, no weight tying, and uses a very small batch size of 8.

## Run 1: Advanced Architecture + BPE Tokenizer (500 steps)
- **Hypothesis**: Introducing a custom BPE tokenizer (vocab 1024) to reduce Hindi sequence length, tying weights, upgrading layers to RMSNorm + RoPE + SwiGLU MLP, and using AdamW + Cosine LR scheduler with warmup + gradient clipping + batch size 16 will improve learning efficiency and lower dev bpb.
- **What Changed**: Custom BPE tokenizer trained on train corpus (compression 1.91x), tied embedding-head weights, RMSNorm (replacing LayerNorm), RoPE positional encoding, SwiGLU MLP (replacing GELU MLP), AdamW optimizer with weight decay = 0.1, linear warmup (200 steps) + cosine LR schedule (peak LR 2e-3), gradient clipping = 1.0, batch size = 16.
- **Steps**: 500
- **Parameters**: 1,967,808
- **Dev bpb**: 1.9637
- **Conclusion**: Exceptional performance improvement. The model achieves 1.9637 dev bpb in just 500 steps (compared to 2.3718 for the 2,000-step baseline). The compression from the tokenizer and efficiency of weight-tied RoPE/RMSNorm/SwiGLU allows the model to learn much faster.

## Run 2: Higher Learning Rate (500 steps)
- **Hypothesis**: Increasing the learning rate from 2e-3 to 3e-3 will accelerate convergence on our custom architecture.
- **What Changed**: Learning rate set to 3e-3. Other settings identical to Run 1.
- **Steps**: 500
- **Parameters**: 1,967,808
- **Dev bpb**: 1.9843
- **Conclusion**: Worse convergence. Step 500 training loss was 2.5266 (vs 2.4934 in Run 1) and dev bpb was 1.9843 (vs 1.9637 in Run 1). The learning rate of 3e-3 is slightly too high for this configuration, leading to sub-optimal updates.

## Run 3: Lower Learning Rate (500 steps)
- **Hypothesis**: Decreasing the learning rate from 2e-3 to 1.5e-3 will provide smoother and more optimal parameter updates, leading to a better final dev loss.
- **What Changed**: Learning rate set to 1.5e-3. Other settings identical to Run 1.
- **Steps**: 500
- **Parameters**: 1,967,808
- **Dev bpb**: 1.9888
- **Conclusion**: Slightly slower convergence. Step 500 training loss was 2.5470 (vs 2.4934 in Run 1) and dev bpb was 1.9888 (vs 1.9637 in Run 1). Slower updates hurt final metrics at 500 steps, making 2e-3 the optimal peak learning rate.

## Run 4: Best Configuration (Final 2,000 Steps Run)
- **Hypothesis**: Training the optimal configuration (peak LR 2e-3, batch size 16, custom BPE vocab 1024, SwiGLU, RMSNorm, RoPE, weight tying) for the full 2,000 steps will allow the model to fully converge and achieve the lowest dev bpb.
- **What Changed**: Trained for the full 2,000 steps.
- **Steps**: 2000
- **Parameters**: 1,967,808
- **Dev bpb**: 1.7105
- **Conclusion**: Extremely strong final model performance. The model successfully fully converged, bringing final training loss to 1.9620 and dev bpb down to 1.7105. This represents a huge improvement over the baseline dev bpb of 2.3718.
