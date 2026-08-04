"""Tests for immutable paired-training corpora and schedules."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile

from PIL import Image
import torch

from fine_grain.pretrain_data import (
    CorpusRecord,
    CorpusWriter,
    ConsecutiveModalityBatchSampler,
    PretrainCollator,
    SQLiteVLMData,
    build_schedule,
    stable_validation_split,
    text_continuation_chunks,
    truncate_utf8,
)


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (19, 11), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_utf8_truncation_and_chunks_are_lossless_at_boundaries():
    text = "ab中文cd"
    assert truncate_utf8(text, 5) == "ab中"
    chunks = list(text_continuation_chunks(text, target_bytes=5, prompt_bytes=3))
    assert "".join(target for _, target in chunks) == text
    assert all(len(target.encode("utf-8")) <= 5 for _, target in chunks)


def test_validation_split_is_stable():
    values = [stable_validation_split(f"sample-{index}") for index in range(1000)]
    assert values == [stable_validation_split(f"sample-{index}") for index in range(1000)]
    assert 2 <= values.count("validation") <= 25


def test_corpus_schedule_and_collation_are_deterministic():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "corpus.sqlite"
        schedule_a = Path(directory) / "a.jsonl"
        schedule_b = Path(directory) / "b.jsonl"
        with CorpusWriter(database) as writer:
            for index in range(8):
                writer.add(
                    CorpusRecord(
                        sample_id=f"ocr-{index}",
                        source="fixture",
                        split="train",
                        task="ocr",
                        prompt_text="",
                        target_text=f"text-{index}",
                        image_bytes=image_bytes(),
                        width=19,
                        height=11,
                    )
                )
        metadata_a = build_schedule(database, schedule_a, seed=17)
        metadata_b = build_schedule(database, schedule_b, seed=17)
        assert metadata_a["sha256"] == metadata_b["sha256"]
        assert schedule_a.read_bytes() == schedule_b.read_bytes()

        dataset = SQLiteVLMData(database, schedule_a)
        batch = PretrainCollator(image_size=16)([dataset[0], dataset[1]])
        assert batch.images.shape == (2, 3, 16, 16)
        assert batch.target_tokens == len("text-0".encode()) * 2 + 2
        assert batch.tokens.target_labels.shape == batch.tokens.target_input_ids.shape


def test_collator_rejects_mixed_modalities():
    visual = CorpusRecord("v", "s", "train", "ocr", "", "x", image_bytes())
    text = CorpusRecord("t", "s", "train", "text", "", "x", None)
    try:
        PretrainCollator()([visual, text])
    except ValueError as error:
        assert "mix" in str(error)
    else:
        raise AssertionError("mixed-modality batch must be rejected")


def test_batch_sampler_preserves_order_and_never_mixes_modalities():
    schedule = [
        {"task": task}
        for task in ("ocr", "ocr", "text", "text", "text", "document", "ocr")
    ]
    sampler = ConsecutiveModalityBatchSampler(schedule, batch_size=2)
    batches = list(sampler)
    assert batches == [[0, 1], [2, 3], [4], [5, 6]]
    assert len(sampler) == len(batches)
