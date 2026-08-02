# VLM frontend A/B/C comparison

- Backend: `local_tinylm path=D:\ml_cache\local_tinylm d_llm=256 cache=D:\ml_cache (MS failed: 0 tries)`
- Cache: `D:\ml_cache` (D: drive)
- res=32 patch=4 steps=60 topk=2 ST=on for B/C

| kind | T | final_loss | probe_token_acc | notes |
|------|---|------------|-----------------|-------|
| A | 64 | 5.3738 | 0.075 |  |
| B | 64 | 5.4919 | 0.000 |  PR=42.6 r99=61.8 sup=2.0 |
| C | 48 | 5.3450 | 0.031 | 16+32 PR=14.4 r99=29.9 sup=2.0 |
| A | 32 | 5.3848 | 0.056 |  |
| B | 32 | 5.4349 | 0.037 |  PR=14.8 r99=29.5 sup=2.0 |
| C | 32 | 5.2378 | 0.037 | 16+16 PR=7.2 r99=15.0 sup=2.0 |
| A | 16 | 5.4446 | 0.056 |  |
| B | 16 | 5.5276 | 0.000 |  PR=7.2 r99=15.1 sup=2.0 |
| C | 12 | 5.2861 | 0.025 | 4+8 PR=4.4 r99=7.0 sup=2.0 |

## Short conclusion

- Best probe accuracy by kind (this short run): A=T64 acc=0.075, B=T32 acc=0.037, C=T32 acc=0.037
- B token compression: T64:0.000 → T32:0.037 → T16:0.000
- Slice uses ST + fixed topk write; effective-slot metrics logged when available.
- Short steps / synthetic mix only — not a full LLaVA claim; use for ranking frontends.
