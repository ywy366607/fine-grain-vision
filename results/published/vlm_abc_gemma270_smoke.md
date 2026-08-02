# VLM frontend A/B/C comparison

- Backend: `local-D path=D:\ml_cache\huggingface\models\google--gemma-3-270m d_llm=640 dtype=torch.float32 cache=D:\ml_cache`
- Cache: `D:\ml_cache` (D: drive)
- res=32 patch=4 steps=15 topk=2 ST=on for B/C

| kind | T | final_loss | probe_token_acc | notes |
|------|---|------------|-----------------|-------|
| A | 64 | 4.3291 | 0.407 |  |
| B | 64 | 5.6804 | 0.012 |  PR=49.3 r99=62.8 sup=2.0 |
| C | 48 | 6.1853 | 0.089 | 16+32 PR=20.5 r99=31.2 sup=2.0 |
| A | 32 | 4.4882 | 0.444 |  |
| B | 32 | 6.3619 | 0.052 |  PR=16.4 r99=30.2 sup=2.0 |
| C | 32 | 6.8054 | 0.089 | 16+16 PR=11.4 r99=15.8 sup=2.0 |

## Short conclusion

- Best probe accuracy by kind (this short run): A=T32 acc=0.444, C=T48 acc=0.089, B=T32 acc=0.052
- B token compression: T64:0.012 → T32:0.052
- Slice uses ST + fixed topk write; effective-slot metrics logged when available.
- Short steps / synthetic mix only — not a full LLaVA claim; use for ranking frontends.
