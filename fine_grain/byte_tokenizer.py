"""A minimal UTF-8 byte tokenizer for from-scratch multimodal pretraining."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BYTE_OFFSET = 3
OCR_ID = 259
DOCUMENT_ID = 260
TEXT_ID = 261
IMAGE_START_ID = 262
IMAGE_END_ID = 263
VOCAB_SIZE = 264


@dataclass
class TeacherForcingBatch:
    """Padded prompt and right-shifted target tensors."""

    prompt_ids: torch.Tensor
    prompt_mask: torch.Tensor
    target_input_ids: torch.Tensor
    target_labels: torch.Tensor
    target_mask: torch.Tensor


class ByteTokenizer:
    """Lossless UTF-8 tokenizer with a small fixed multimodal vocabulary."""

    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, add_eos: bool = False) -> list[int]:
        ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        values = []
        for token in ids:
            token = int(token)
            if token == EOS_ID:
                break
            if BYTE_OFFSET <= token < BYTE_OFFSET + 256:
                values.append(token - BYTE_OFFSET)
        return bytes(values).decode("utf-8", errors="replace")

    @staticmethod
    def task_prompt(task: str) -> list[int]:
        tasks = {
            "ocr": OCR_ID,
            "document": DOCUMENT_ID,
            "text": TEXT_ID,
        }
        try:
            task_id = tasks[task]
        except KeyError as error:
            raise ValueError(f"unknown task: {task}") from error
        return [BOS_ID, task_id]

    def teacher_forcing_batch(
        self,
        prompts: list[list[int]],
        targets: list[str],
        *,
        device: torch.device | str | None = None,
    ) -> TeacherForcingBatch:
        if len(prompts) != len(targets) or not prompts:
            raise ValueError("prompts and targets must have the same nonzero length")

        encoded_targets = [self.encode(text, add_eos=True) for text in targets]
        target_inputs = [[BOS_ID, *ids[:-1]] for ids in encoded_targets]
        prompt_ids, prompt_mask = self._pad(prompts, PAD_ID, device=device)
        target_input_ids, target_mask = self._pad(target_inputs, PAD_ID, device=device)
        target_labels, _ = self._pad(encoded_targets, -100, device=device)
        return TeacherForcingBatch(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            target_input_ids=target_input_ids,
            target_labels=target_labels,
            target_mask=target_mask,
        )

    @staticmethod
    def _pad(
        rows: list[list[int]],
        fill: int,
        *,
        device: torch.device | str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(len(row) for row in rows)
        values = torch.full((len(rows), width), fill, dtype=torch.long, device=device)
        mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
        for index, row in enumerate(rows):
            if row:
                values[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
                mask[index, : len(row)] = True
        return values, mask
