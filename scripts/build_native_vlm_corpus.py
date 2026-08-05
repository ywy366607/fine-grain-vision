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
from typing import Iterable, Iterator

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


def encode_training_image(image: object, maximum_side: int) -> tuple[bytes, int, int]:
    """Bound storage while retaining substantially more detail than training crops."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            image = io.BytesIO(image["bytes"])
        elif image.get("path"):
            image = image["path"]
        else:
            raise ValueError("image payload has neither bytes nor path")
    with Image.open(image) if not isinstance(image, Image.Image) else image as opened:
        prepared = opened.convert("RGB")
        prepared.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        prepared.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue(), prepared.width, prepared.height


def parse_source_budgets(values: list[str]) -> list[tuple[str, int]]:
    parsed = []
    for value in values:
        try:
            name, budget_text = value.rsplit("=", 1)
            budget = int(budget_text.replace("_", ""))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"expected CONFIG=TARGET_BYTES, received {value!r}"
            ) from error
        if not name or budget < 1:
            raise argparse.ArgumentTypeError(
                f"invalid source budget {value!r}"
            )
        parsed.append((name, budget))
    return parsed


def source_train_tokens(writer: CorpusWriter, source: str) -> int:
    row = writer.connection.execute(
        "SELECT COALESCE(SUM(target_bytes), 0) FROM samples "
        "WHERE source=? AND split='train'",
        (source,),
    ).fetchone()
    return int(row[0])


def import_existing_source(
    writer: CorpusWriter,
    database: str,
    *,
    source: str,
    budget: int,
) -> dict[str, int]:
    """Migrate a bounded source from a v1 corpus into normalized schema v2."""
    train_tokens = source_train_tokens(writer, source)
    inserted = 0
    connection = sqlite3.connect(
        f"file:{Path(database).resolve()}?mode=ro", uri=True
    )
    columns = {row[1] for row in connection.execute("PRAGMA table_info(samples)")}
    if "image_id" in columns:
        rows = connection.execute(
            "SELECT s.sample_id, s.source, s.split, s.task, s.prompt_text, "
            "s.target_text, COALESCE(s.image, i.payload), "
            "COALESCE(s.width, i.width), COALESCE(s.height, i.height) "
            "FROM samples AS s LEFT JOIN images AS i ON i.image_id=s.image_id "
            "WHERE s.source=? ORDER BY s.sample_id",
            (source,),
        )
    else:
        rows = connection.execute(
            "SELECT sample_id, source, split, task, prompt_text, target_text, "
            "image, width, height FROM samples WHERE source=? ORDER BY sample_id",
            (source,),
        )
    for row in rows:
        record = CorpusRecord(*row[:6], image_bytes=row[6], width=row[7], height=row[8])
        if record.split == "train":
            target = truncate_utf8(
                record.target_text, max(budget - train_tokens - 1, 0)
            )
            if not target:
                break
            record = CorpusRecord(
                **{**record.__dict__, "target_text": target}
            )
        if writer.add(record):
            inserted += 1
            if record.split == "train":
                train_tokens += record.target_bytes
        if inserted and inserted % 5000 == 0:
            writer.commit()
        if train_tokens >= budget:
            break
    connection.close()
    writer.commit()
    return {"inserted": inserted, "train_tokens": train_tokens}


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


def add_qa_rows(
    writer: CorpusWriter,
    *,
    source: str,
    rows: Iterable[tuple[str, object, Iterable[tuple[str, str]]]],
    budget: int,
    maximum_target_bytes: int,
    maximum_image_side: int,
) -> dict[str, int]:
    """Store one image once while flattening its independent question-answer pairs."""
    train_tokens = source_train_tokens(writer, source)
    inserted = failed = skipped_multi_image = 0
    existing_sample_ids = {
        row[0]
        for row in writer.connection.execute(
            "SELECT sample_id FROM samples WHERE source=?", (source,)
        )
    }
    for image_key, image, conversations in rows:
        split = stable_validation_split(f"{source}:{image_key}")
        prepared = []
        reserved_train_tokens = train_tokens
        for qa_index, (prompt, answer) in enumerate(conversations):
            sample_id = f"{source}:{image_key}:{qa_index:04d}"
            if sample_id in existing_sample_ids:
                continue
            limit = maximum_target_bytes
            if split == "train":
                limit = min(limit, max(budget - reserved_train_tokens - 1, 0))
            target = truncate_utf8(answer or "", limit)
            if target:
                prepared.append((sample_id, prompt or "", target))
                if split == "train":
                    reserved_train_tokens += len(target.encode("utf-8")) + 1
            if split == "train" and reserved_train_tokens >= budget:
                break
        if not prepared:
            if train_tokens >= budget:
                break
            continue
        try:
            payload, width, height = encode_training_image(image, maximum_image_side)
        except Exception as error:
            failed += 1
            print(f"invalid image {source}:{image_key}: {error}", flush=True)
            continue
        for sample_id, prompt, target in prepared:
            record = CorpusRecord(
                sample_id=sample_id,
                source=source,
                split=split,
                task="document",
                prompt_text=prompt,
                target_text=target,
                image_bytes=payload,
                width=width,
                height=height,
            )
            if writer.add(record):
                inserted += 1
                if split == "train":
                    train_tokens += record.target_bytes
            if train_tokens >= budget:
                break
        if inserted and inserted % 2000 == 0:
            writer.commit()
            print(
                f"{source} inserted={inserted} train_tokens={train_tokens}",
                flush=True,
            )
        if train_tokens >= budget:
            break
    writer.commit()
    return {
        "inserted": inserted,
        "failed": failed,
        "skipped_multi_image": skipped_multi_image,
        "train_tokens": train_tokens,
    }


def cauldron_rows(config: str, revision: str) -> Iterator[
    tuple[str, object, Iterable[tuple[str, str]]]
]:
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceM4/the_cauldron",
        name=config,
        split="train",
        revision=revision,
        streaming=True,
        batch_size=8,
    ).decode(False)
    for row_index, row in enumerate(dataset):
        images = row.get("images") or []
        if len(images) != 1:
            continue
        conversations = (
            (turn.get("user", ""), turn.get("assistant", ""))
            for turn in (row.get("texts") or [])
        )
        yield f"{row_index:09d}", images[0], conversations


def pixmo_docs_rows(config: str, revision: str) -> Iterator[
    tuple[str, object, Iterable[tuple[str, str]]]
]:
    os.environ.pop("ALL_PROXY", None)
    os.environ.pop("all_proxy", None)
    from datasets import load_dataset

    dataset = load_dataset(
        "allenai/pixmo-docs",
        name=config,
        split="train",
        revision=revision,
        streaming=True,
        batch_size=8,
    ).decode(False)
    for row_index, row in enumerate(dataset):
        questions = row.get("questions") or {}
        conversations = zip(
            questions.get("question") or [], questions.get("answer") or []
        )
        yield str(row.get("image_id") or f"{row_index:09d}"), row["image"], conversations


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
        "SELECT s.source, s.split, COUNT(*), SUM(s.target_bytes), "
        "COUNT(DISTINCT s.image_id), "
        "COALESCE((SELECT SUM(LENGTH(i.payload)) FROM images AS i WHERE i.image_id IN "
        "(SELECT DISTINCT q.image_id FROM samples AS q "
        "WHERE q.source=s.source AND q.split=s.split)), SUM(LENGTH(s.image)), 0) "
        "FROM samples AS s GROUP BY s.source, s.split ORDER BY s.source, s.split"
    ).fetchall()
    image_row = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(LENGTH(payload)), 0) FROM images"
    ).fetchone()
    connection.close()
    return {
        "unique_images": image_row[0],
        "stored_image_bytes": image_row[1],
        "sources": [
            {
                "source": source,
                "split": split,
                "samples": count,
                "target_tokens": tokens,
                "unique_images": unique_images,
                "compressed_image_bytes": image_bytes or 0,
            }
            for source, split, count, tokens, unique_images, image_bytes in rows
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True)
    parser.add_argument("--recipe")
    parser.add_argument("--existing-database")
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
    parser.add_argument("--maximum-image-side", type=int, default=1536)
    parser.add_argument(
        "--cauldron-config",
        action="append",
        default=[],
        metavar="CONFIG=TARGET_BYTES",
    )
    parser.add_argument(
        "--cauldron-revision",
        default="847a98a779b1652d65111daf20c972dfcd333605",
    )
    parser.add_argument(
        "--pixmo-docs-config",
        action="append",
        default=[],
        metavar="CONFIG=TARGET_BYTES",
    )
    parser.add_argument(
        "--pixmo-docs-revision",
        default="d887597bf4af2bc61a4210071a8cef898287e6fb",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    recipe = {}
    if args.recipe:
        recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    cauldron_specs = [
        *args.cauldron_config,
        *[
            f"{item['config']}={item['target_bytes']}"
            for item in recipe.get("cauldron", [])
        ],
    ]
    pixmo_docs_specs = [
        *args.pixmo_docs_config,
        *[
            f"{item['config']}={item['target_bytes']}"
            for item in recipe.get("pixmo_docs", [])
        ],
    ]
    cauldron_revision = recipe.get(
        "cauldron_revision", args.cauldron_revision
    )
    pixmo_docs_revision = recipe.get(
        "pixmo_docs_revision", args.pixmo_docs_revision
    )
    existing_specs = recipe.get("existing", [])
    if existing_specs and not args.existing_database:
        parser.error("the recipe contains existing sources; pass --existing-database")

    actions = {}
    with CorpusWriter(args.database) as writer:
        for item in existing_specs:
            source = item["source"]
            actions[f"existing_{source}"] = import_existing_source(
                writer,
                args.existing_database,
                source=source,
                budget=int(item["target_bytes"]),
            )
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
        for config, budget in parse_source_budgets(cauldron_specs):
            source = f"cauldron_{config}"
            actions[source] = add_qa_rows(
                writer,
                source=source,
                rows=cauldron_rows(config, cauldron_revision),
                budget=budget,
                maximum_target_bytes=args.max_target_bytes,
                maximum_image_side=args.maximum_image_side,
            )
        for config, budget in parse_source_budgets(pixmo_docs_specs):
            source = f"pixmo_docs_{config}"
            actions[source] = add_qa_rows(
                writer,
                source=source,
                rows=pixmo_docs_rows(config, pixmo_docs_revision),
                budget=budget,
                maximum_target_bytes=args.max_target_bytes,
                maximum_image_side=args.maximum_image_side,
            )
        writer.set_metadata("build_arguments", vars(args))
        writer.set_metadata(
            "resolved_public_recipe",
            {
                "cauldron_revision": cauldron_revision,
                "cauldron": cauldron_specs,
                "pixmo_docs_revision": pixmo_docs_revision,
                "pixmo_docs": pixmo_docs_specs,
                "existing": existing_specs,
            },
        )
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
