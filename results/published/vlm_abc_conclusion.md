# VLM frontend A/B/C comparison

- Backend: `local_tinylm` on `D:\ml_cache\local_tinylm` (ModelScope Gemma/Pythia unavailable; HF blocked; **no C: cache**)
- Protocol: res=32, patch=4 (native patch tokens=**64**), steps=50, batch=4, topk=2, **ST=on** for B/C
- Probe: teacher-forced token acc with **correct causal alignment**  
  (`pred = logits[:, T-1:T+L-1].argmax` vs `ids`, mask>0) — see `tests/test_probe_alignment.py`
- Entry: `python scripts/train_vlm_frontends.py --mode sweep --prefer local --T_list 64 32 16 --amp`

| kind | T | final_loss ↓ | probe_token_acc ↑ | notes |
|------|---|--------------|-------------------|-------|
| A | 64 | 5.461 | **0.271** | pure patch (patch-matched T) |
| B | 64 | 5.686 | 0.167 | pure slice; PR≈43 r99≈62 **sup=2.0** |
| C | 48 | 5.456 | 0.177 | budget **16+32** |
| A | 32 | 5.414 | **0.271** | compressed patch |
| B | 32 | 5.605 | 0.167 | slice T=32; **sup=2.0** |
| C | 32 | **5.337** | 0.250 | budget **16+16** (best loss) |
| A | 16 | 5.453 | **0.271** | |
| B | 16 | 5.587 | 0.167 | |
| C | 12 | 5.503 | 0.167 | budget **4+8** |

## Short conclusion

1. **Pipeline works on 4GB**: A/B/C all train; losses decrease; visual tokens are `T × d_llm` into a frozen LM.
2. **B is not locked tiny**: T=64 matches patch token count, then compresses 64→32→16; slot metrics show **deslice support=2** (topk).
3. **After probe fix**: **A** leads probe token-acc (~0.27); **C@T32 (16+16)** best final_loss (~5.34) and competitive probe (~0.25); **B** trails on this short synthetic + tinylm setup but loss still falls with T.
4. **B compression curve (probe)**: T64=0.167 → T32=0.167 → T16=0.167 (flat here; needs longer train / real LLM to separate).
5. Short steps / synthetic mix only — ranking frontends, not SOTA VLM claims.

Reproduce:

```bash
set ML_CACHE_ROOT=D:\ml_cache
python tests/test_probe_alignment.py
python tests/test_vlm_frontends.py
python scripts/train_vlm_frontends.py --mode sweep --prefer local --res 32 --T_list 64 32 16 --steps 50 --amp
```
