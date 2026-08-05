# Native VLM Corpus V2

## Objective

The corpus must train image-conditioned behavior rather than merely minimize
teacher-forced language loss. Every visual source is stored as independent
`image + prompt -> answer` records. Questions never appear in the supervised
target. All question-answer records sharing an image reference one deduplicated
image blob in schema v2.

## Registered Public Sources

The primary source is
[`HuggingFaceM4/the_cauldron`](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
at commit `847a98a779b1652d65111daf20c972dfcd333605`. The registered configs cover:

- OCR and documents: Rendered diagrams, InfographicVQA, DocVQA, OCR-VQA,
  TextVQA, ST-VQA, and IAM;
- charts and diagrams: ChartQA, PlotQA, FigureQA, DVQA, and AI2D;
- general perception: TallyQA, VQAv2, and Localized Narratives;
- screens and visual code: Screen2Words, WebSight, and DaTikZ.

The exact caps are in `configs/native_vlm_corpus_v2.json`. They total 82.1M
visual target bytes before source shortages. The selected remote parquet files
occupy approximately 78 GB. Existing local LaTeX OCR, olmOCR, and SCUT data are
retained, while pure text is capped at 8M target bytes.

For the architecture gate, `configs/native_vlm_corpus_gate_v2.json` is the
registered first run. It retains the bounded local sources and adds only 5.5M
bytes of short-answer chart, counting, and diagram supervision. This keeps the
whole gate near 27M target bytes. The 82.1M recipe is archival until the compact
run demonstrates positive image causality.

[`allenai/pixmo-docs`](https://huggingface.co/datasets/allenai/pixmo-docs)
at commit `d887597bf4af2bc61a4210071a8cef898287e6fb` is an audited fallback for
charts, diagrams, tables, and synthetic documents. It is not in the primary
mixture because the Cauldron already covers those tasks and downloading both
would duplicate supervision.

## Rejected Sources

- Cauldron `clevr_math` references private absolute image paths and is not
  reproducibly usable from the public repository.
- LLaVA-OneVision Mid Data is restricted by its card to academic research and
  education, which is unsuitable for unrestricted public model weights.
- The original 12.9 TB RenderedText collection has no declared dataset license.
  Local programmatic rendering is safer and gives exact target provenance.
- Docmatix is valid and MIT-licensed, but its 982 GB image form is unnecessary
  for the first mechanism run. Add a bounded shard sample only after the
  Cauldron gate passes.

## Acceptance Gates

1. At 2M target tokens, matched image NLL must beat both blank and shuffled on
   a held-out, strictly paired visual set.
2. At 8M tokens, every major family must show a positive paired margin; aggregate
   language NLL cannot substitute for this test.
3. Only a run passing both gates may continue to the full budget and enter the
   Slice-versus-Patch comparison.

The public Cauldron combines datasets with source-specific licenses. Before a
model release, retain attribution for every imported config and complete the
source-level license review described by its dataset card.
