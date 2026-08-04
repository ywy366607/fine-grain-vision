"""Deterministic corpus and batching primitives for paired VLM pretraining."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler

from fine_grain.byte_tokenizer import ByteTokenizer, TeacherForcingBatch


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorpusRecord:
    sample_id: str
    source: str
    split: str
    task: str
    prompt_text: str
    target_text: str
    image_bytes: bytes | None
    width: int | None = None
    height: int | None = None

    @property
    def target_bytes(self) -> int:
        return len(self.target_text.encode("utf-8")) + 1  # EOS is loss-bearing.


@dataclass
class PretrainBatch:
    tokens: TeacherForcingBatch
    images: torch.Tensor | None
    sample_ids: list[str]
    sources: list[str]
    tasks: list[str]
    target_tokens: int


def stable_validation_split(sample_id: str, validation_percent: int = 1) -> str:
    if not 0 <= validation_percent < 100:
        raise ValueError("validation_percent must be in [0, 100)")
    bucket = int.from_bytes(
        hashlib.blake2b(sample_id.encode("utf-8"), digest_size=8).digest(), "big"
    ) % 100
    return "validation" if bucket < validation_percent else "train"


def truncate_utf8(text: str, maximum_bytes: int) -> str:
    """Keep a valid UTF-8 prefix no longer than maximum_bytes."""
    if maximum_bytes < 1:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def text_continuation_chunks(
    text: str,
    *,
    target_bytes: int,
    prompt_bytes: int,
) -> Iterable[tuple[str, str]]:
    """Yield non-overlapping UTF-8 target chunks with a preceding context window."""
    encoded = text.encode("utf-8")
    start = 0
    while start < len(encoded):
        end = min(start + target_bytes, len(encoded))
        target = encoded[start:end].decode("utf-8", errors="ignore")
        consumed = len(target.encode("utf-8"))
        if not consumed:
            start += 1
            continue
        context_start = max(0, start - prompt_bytes)
        prompt = encoded[context_start:start].decode("utf-8", errors="ignore")
        yield prompt, target
        start += consumed


class CorpusWriter:
    """Transactional writer with source accounting and sample de-duplication."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples (
                sample_id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                split TEXT NOT NULL CHECK(split IN ('train', 'validation')),
                task TEXT NOT NULL CHECK(task IN ('ocr', 'document', 'text')),
                image BLOB,
                prompt_text TEXT NOT NULL,
                target_text TEXT NOT NULL,
                target_bytes INTEGER NOT NULL,
                width INTEGER,
                height INTEGER
            );
            CREATE INDEX IF NOT EXISTS samples_split_source
                ON samples(split, source, sample_id);
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )

    def add(self, record: CorpusRecord) -> bool:
        if not record.target_text:
            return False
        cursor = self.connection.execute(
            """
            INSERT OR IGNORE INTO samples(
                sample_id, source, split, task, image, prompt_text, target_text,
                target_bytes, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.sample_id,
                record.source,
                record.split,
                record.task,
                record.image_bytes,
                record.prompt_text,
                record.target_text,
                record.target_bytes,
                record.width,
                record.height,
            ),
        )
        return cursor.rowcount == 1

    def set_metadata(self, key: str, value: object) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
            (key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
        )

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "CorpusWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def build_schedule(
    database: str | Path,
    output: str | Path,
    *,
    seed: int,
    split: str = "train",
) -> dict[str, object]:
    """Write an immutable shuffled row schedule shared by both model arms."""
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT rowid, sample_id, source, task, target_bytes "
        "FROM samples WHERE split=? ORDER BY source, sample_id",
        (split,),
    ).fetchall()
    connection.close()
    random.Random(seed).shuffle(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    totals: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for rowid, sample_id, source, task, target_count in rows:
            item = {
                "rowid": rowid,
                "sample_id": sample_id,
                "source": source,
                "task": task,
                "target_tokens": target_count,
            }
            line = json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            totals[source] = totals.get(source, 0) + target_count
    return {
        "path": str(output),
        "sha256": digest.hexdigest(),
        "samples": len(rows),
        "target_tokens": sum(totals.values()),
        "source_target_tokens": totals,
        "seed": seed,
        "split": split,
    }


class SQLiteVLMData(Dataset[CorpusRecord]):
    """Read samples in an explicit schedule; each worker owns its DB connection."""

    def __init__(self, database: str | Path, schedule: str | Path) -> None:
        self.database = str(Path(database).resolve())
        with Path(schedule).open(encoding="utf-8") as handle:
            self.schedule = [json.loads(line) for line in handle if line.strip()]
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return len(self.schedule)

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                f"file:{self.database}?mode=ro", uri=True, check_same_thread=False
            )
            self._connection.execute("PRAGMA query_only=ON")
        return self._connection

    def __getitem__(self, index: int) -> CorpusRecord:
        scheduled = self.schedule[index]
        row = self._db().execute(
            "SELECT sample_id, source, split, task, prompt_text, target_text, "
            "image, width, height FROM samples WHERE rowid=?",
            (scheduled["rowid"],),
        ).fetchone()
        if row is None or row[0] != scheduled["sample_id"]:
            raise RuntimeError("corpus no longer matches the immutable schedule")
        return CorpusRecord(*row[:6], image_bytes=row[6], width=row[7], height=row[8])


class ConsecutiveModalityBatchSampler(Sampler[list[int]]):
    """Batch adjacent scheduled samples without mixing text-only and vision."""

    def __init__(
        self,
        schedule: Sequence[dict[str, object]],
        *,
        batch_size: int,
        start_index: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0 <= start_index <= len(schedule):
            raise ValueError("start_index is outside the schedule")
        self.schedule = schedule
        self.batch_size = batch_size
        self.start_index = start_index

    def __iter__(self):
        batch: list[int] = []
        modality: bool | None = None
        for index in range(self.start_index, len(self.schedule)):
            is_visual = self.schedule[index]["task"] != "text"
            if batch and (is_visual != modality or len(batch) == self.batch_size):
                yield batch
                batch = []
            if not batch:
                modality = is_visual
            batch.append(index)
        if batch:
            yield batch

    def __len__(self) -> int:
        count = 0
        current = 0
        modality = None
        for index in range(self.start_index, len(self.schedule)):
            is_visual = self.schedule[index]["task"] != "text"
            if current and (is_visual != modality or current == self.batch_size):
                count += 1
                current = 0
            if not current:
                modality = is_visual
            current += 1
        return count + int(current > 0)


def _letterbox(image_bytes: bytes, image_size: int) -> torch.Tensor:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), "white")
        left = (image_size - image.width) // 2
        top = (image_size - image.height) // 2
        canvas.paste(image, (left, top))
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class PretrainCollator:
    def __init__(self, *, image_size: int = 256) -> None:
        self.image_size = image_size
        self.tokenizer = ByteTokenizer()

    def __call__(self, records: Sequence[CorpusRecord]) -> PretrainBatch:
        if not records:
            raise ValueError("cannot collate an empty batch")
        has_images = [record.image_bytes is not None for record in records]
        if any(has_images) and not all(has_images):
            raise ValueError("a batch cannot mix visual and text-only samples")
        prompts = [
            [
                *self.tokenizer.task_prompt(record.task),
                *self.tokenizer.encode(record.prompt_text),
            ]
            for record in records
        ]
        tokens = self.tokenizer.teacher_forcing_batch(
            prompts, [record.target_text for record in records]
        )
        images = None
        if all(has_images):
            images = torch.stack(
                [_letterbox(record.image_bytes, self.image_size) for record in records]
            )
        return PretrainBatch(
            tokens=tokens,
            images=images,
            sample_ids=[record.sample_id for record in records],
            sources=[record.source for record in records],
            tasks=[record.task for record in records],
            target_tokens=int(tokens.target_mask.sum()),
        )
