#!/usr/bin/env python3
"""Build the immutable paired-training corpus from bounded public sources."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile
from typing import Iterable

from PIL import Image
import pyarrow.parquet as pq

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fine_grain.pretrain_data import (  # noqa: E402
    CorpusRecord,
    CorpusWriter,
    build_schedule,
    stable_validation_split,
    text_continuation_chunks,
    truncate_utf8,
)


def image_payload(value: object) -> bytes:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return value["bytes"]
        if value.get("path"):
            return Path(value["path"]).read_bytes()
    if isinstance(value, bytes):
        return value
    raise ValueError("unsupported parquet image payload")


def image_dimensions(payload: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        return image.width, image.height


def source_train_tokens(writer: CorpusWriter, source: str) -> int:
    row = writer.connection.execute(
        "SELECT COALESCE(SUM(target_bytes), 0) FROM samples "
        "WHERE source=? AND split='train'",
        (source,),
    ).fetchone()
    return int(row[0])


def add_parquet_ocr(
    writer: CorpusWriter,
    path: str,
    *,
    source: str,
    task: str,
    text_column: str,
    budget: int,
    maximum_target_bytes: int,
) -> dict[str, int]:
    train_tokens = source_train_tokens(writer, source)
    inserted = failed = 0
    existing_sample_ids = {
        row[0]
        for row in writer.connection.execute(
            "SELECT sample_id FROM samples WHERE source=?", (source,)
        )
    }
    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(
        batch_size=256, columns=["image", text_column]
    ):
        images = batch.column(0).to_pylist()
        texts = batch.column(1).to_pylist()
        for image, text in zip(images, texts):
            sample_id = f"{source}:{row_index:09d}"
            row_index += 1
            if sample_id in existing_sample_ids:
                continue
            if isinstance(text, list):
                text = "\n".join(part for part in text if part)
            text = text or ""
            split = stable_validation_split(sample_id)
            remaining = budget - train_tokens
            limit = maximum_target_bytes
            if split == "train":
                limit = min(limit, max(remaining - 1, 0))
            target = truncate_utf8(text, limit)
            if not target:
                continue
            try:
                payload = image_payload(image)
                width, height = image_dimensions(payload)
            except Exception as error:
                failed += 1
                print(f"invalid image {sample_id}: {error}", flush=True)
                continue
            record = CorpusRecord(
                sample_id=sample_id,
                source=source,
                split=split,
                task=task,
                prompt_text="",
                target_text=target,
                image_bytes=payload,
                width=width,
                height=height,
            )
            if writer.add(record):
                inserted += 1
                if split == "train":
                    train_tokens += record.target_bytes
            if inserted and inserted % 1000 == 0:
                writer.commit()
            if train_tokens >= budget:
                writer.commit()
                return {
                    "inserted": inserted,
                    "failed": failed,
                    "train_tokens": train_tokens,
                }
    writer.commit()
    return {
        "inserted": inserted,
        "failed": failed,
        "train_tokens": train_tokens,
    }


def add_fineweb(
    writer: CorpusWriter,
    *,
    budget: int,
    maximum_target_bytes: int,
    prompt_bytes: int,
) -> dict[str, int]:
    source = "fineweb_edu"
    train_tokens = source_train_tokens(writer, source)
    inserted = 0
    # HTTPX otherwise prefers an unsupported SOCKS override over the working
    # HTTP proxy configured on this machine.
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for document_index, example in enumerate(dataset):
        document_id = str(example.get("id") or document_index)
        for chunk_index, (prompt, target) in enumerate(
            text_continuation_chunks(
                example["text"],
                target_bytes=maximum_target_bytes,
                prompt_bytes=prompt_bytes,
            )
        ):
            sample_id = f"{source}:{document_id}:{chunk_index:04d}"
            split = stable_validation_split(sample_id)
            if split == "train":
                remaining = budget - train_tokens
                target = truncate_utf8(target, max(remaining - 1, 0))
            if not target:
                continue
            record = CorpusRecord(
                sample_id=sample_id,
                source=source,
                split=split,
                task="text",
                prompt_text=prompt,
                target_text=target,
                image_bytes=None,
            )
            if writer.add(record):
                inserted += 1
                if split == "train":
                    train_tokens += record.target_bytes
            if inserted and inserted % 2000 == 0:
                writer.commit()
            if train_tokens >= budget:
                writer.commit()
                return {"inserted": inserted, "train_tokens": train_tokens}
    writer.commit()
    return {"inserted": inserted, "train_tokens": train_tokens}


def render_pdf(pdf: bytes, scale_to: int) -> tuple[bytes, int, int]:
    process = subprocess.run(
        [
            "pdftocairo",
            "-f",
            "1",
            "-l",
            "1",
            "-singlefile",
            "-scale-to",
            str(scale_to),
            "-jpeg",
            "-jpegopt",
            "quality=90",
            "-",
            "-",
        ],
        input=pdf,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode or not process.stdout:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace")[:500])
    width, height = image_dimensions(process.stdout)
    return process.stdout, width, height


def select_olm_records(
    parquet_path: str,
    *,
    source: str,
    existing_train_tokens: int,
    budget: int,
    maximum_target_bytes: int,
    existing_sample_ids: set[str],
) -> dict[str, CorpusRecord]:
    selected: dict[str, CorpusRecord] = {}
    train_tokens = existing_train_tokens
    parquet = pq.ParquetFile(parquet_path)
    columns = ["id", "pdf_relpath", "natural_text"]
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        for row in batch.to_pylist():
            sample_id = f"{source}:{row['id']}"
            if sample_id in existing_sample_ids:
                continue
            split = stable_validation_split(sample_id)
            limit = maximum_target_bytes
            if split == "train":
                limit = min(limit, max(budget - train_tokens - 1, 0))
            target = truncate_utf8(row["natural_text"] or "", limit)
            if not target:
                continue
            member = row["pdf_relpath"].split(":", 1)[-1]
            record = CorpusRecord(
                sample_id=sample_id,
                source=source,
                split=split,
                task="document",
                prompt_text="",
                target_text=target,
                image_bytes=None,
            )
            selected[member] = record
            if split == "train":
                train_tokens += record.target_bytes
            if train_tokens >= budget:
                return selected
    return selected


def add_olmocr(
    writer: CorpusWriter,
    parquet_path: str,
    tar_path: str,
    *,
    budget: int,
    maximum_target_bytes: int,
    workers: int,
    render_size: int,
) -> dict[str, int]:
    source = "olmocr_mix"
    train_tokens = source_train_tokens(writer, source)
    existing_sample_ids = {
        row[0]
        for row in writer.connection.execute(
            "SELECT sample_id FROM samples WHERE source=?", (source,)
        )
    }
    needed = select_olm_records(
        parquet_path,
        source=source,
        existing_train_tokens=train_tokens,
        budget=budget,
        maximum_target_bytes=maximum_target_bytes,
        existing_sample_ids=existing_sample_ids,
    )
    inserted = failed = 0
    pending: dict[Future, CorpusRecord] = {}

    def consume(done: Iterable[Future]) -> None:
        nonlocal inserted, failed, train_tokens
        for future in done:
            record = pending.pop(future)
            try:
                payload, width, height = future.result()
            except Exception as error:
                failed += 1
                print(f"render failed {record.sample_id}: {error}", flush=True)
                continue
            complete = CorpusRecord(
                **{
                    **record.__dict__,
                    "image_bytes": payload,
                    "width": width,
                    "height": height,
                }
            )
            if writer.add(complete):
                inserted += 1
                if complete.split == "train":
                    train_tokens += complete.target_bytes
            if inserted and inserted % 250 == 0:
                writer.commit()
                print(
                    f"olmOCR inserted={inserted} train_tokens={train_tokens} "
                    f"remaining={len(needed)}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive:
                record = needed.pop(member.name, None)
                if record is None or not member.isfile():
                    continue
                payload = archive.extractfile(member).read()
                pending[executor.submit(render_pdf, payload, render_size)] = record
                if len(pending) >= workers * 2:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    consume(done)
                if not needed:
                    break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            consume(done)
    writer.commit()
    missing = len(needed)
    if missing:
        print(
            f"olmOCR asset shortage: {missing} selected PDFs were absent from "
            f"{tar_path}; retained all successfully rendered unique samples",
            flush=True,
        )
    return {
        "inserted": inserted,
        "failed": failed,
        "missing": missing,
        "train_tokens": train_tokens,
    }


def corpus_report(database: str) -> dict[str, object]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT source, split, COUNT(*), SUM(target_bytes), SUM(LENGTH(image)) "
        "FROM samples GROUP BY source, split ORDER BY source, split"
    ).fetchall()
    connection.close()
    return {
        "sources": [
            {
                "source": source,
                "split": split,
                "samples": count,
                "target_tokens": tokens,
                "compressed_image_bytes": image_bytes or 0,
            }
            for source, split, count, tokens, image_bytes in rows
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--schedule")
    parser.add_argument("--validation-schedule")
    parser.add_argument("--report")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--latex-parquet")
    parser.add_argument("--latex-budget", type=int, default=16_000_000)
    parser.add_argument("--scut-parquet")
    parser.add_argument("--scut-budget", type=int, default=1_000_000)
    parser.add_argument("--olm-parquet")
    parser.add_argument("--olm-tar")
    parser.add_argument("--olm-budget", type=int, default=23_000_000)
    parser.add_argument("--fineweb", action="store_true")
    parser.add_argument("--fineweb-budget", type=int, default=24_000_000)
    parser.add_argument("--max-target-bytes", type=int, default=512)
    parser.add_argument("--document-target-bytes", type=int, default=1024)
    parser.add_argument("--prompt-bytes", type=int, default=128)
    parser.add_argument("--render-size", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    actions = {}
    with CorpusWriter(args.database) as writer:
        if args.latex_parquet:
            actions["latex_ocr"] = add_parquet_ocr(
                writer,
                args.latex_parquet,
                source="latex_ocr",
                task="ocr",
                text_column="text",
                budget=args.latex_budget,
                maximum_target_bytes=args.max_target_bytes,
            )
        if args.scut_parquet:
            actions["scut_hccdoc"] = add_parquet_ocr(
                writer,
                args.scut_parquet,
                source="scut_hccdoc",
                task="document",
                text_column="texts",
                budget=args.scut_budget,
                maximum_target_bytes=args.document_target_bytes,
            )
        if bool(args.olm_parquet) != bool(args.olm_tar):
            parser.error("--olm-parquet and --olm-tar must be provided together")
        if args.olm_parquet:
            actions["olmocr_mix"] = add_olmocr(
                writer,
                args.olm_parquet,
                args.olm_tar,
                budget=args.olm_budget,
                maximum_target_bytes=args.document_target_bytes,
                workers=args.workers,
                render_size=args.render_size,
            )
        if args.fineweb:
            actions["fineweb_edu"] = add_fineweb(
                writer,
                budget=args.fineweb_budget,
                maximum_target_bytes=args.max_target_bytes,
                prompt_bytes=args.prompt_bytes,
            )
        writer.set_metadata("build_arguments", vars(args))
        writer.set_metadata("actions", actions)

    report = corpus_report(args.database)
    report["actions"] = actions
    if args.schedule:
        report["schedule"] = build_schedule(
            args.database, args.schedule, seed=args.seed
        )
    if args.validation_schedule:
        report["validation_schedule"] = build_schedule(
            args.database,
            args.validation_schedule,
            seed=args.seed,
            split="validation",
        )
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.fineweb:
        # datasets/pyarrow can leave an HTTP streaming worker alive and crash
        # during CPython extension teardown. All transactions and output files
        # are closed above; bypassing interpreter teardown gives a clean,
        # reproducible exit without weakening corpus durability.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
