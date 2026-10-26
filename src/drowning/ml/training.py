"""Shared Stage A/B training and evaluation steps for Kaggle workers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from drowning.ml.model import CLASS_KEYS, DrowningTemporalModel, torch


def image_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[-1] != len(CLASS_KEYS):
        raise ValueError("image logits must have shape N,3")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("image targets must have shape N")
    return torch.nn.functional.cross_entropy(logits, targets.long(), weight=class_weights)


def temporal_loss(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor,
                  class_weights: torch.Tensor | None = None, ignore_index: int = -100) -> torch.Tensor:
    if logits.ndim != 3 or logits.shape[-1] != len(CLASS_KEYS):
        raise ValueError("temporal logits must have shape B,T,3")
    if targets.shape != logits.shape[:2] or valid_mask.shape != logits.shape[:2]:
        raise ValueError("targets and valid_mask must have shape B,T")
    supervision = valid_mask.bool() & targets.ne(ignore_index)
    if not torch.any(supervision):
        raise ValueError("temporal batch contains no supervised valid anchors")
    return torch.nn.functional.cross_entropy(logits[supervision], targets[supervision].long(), weight=class_weights)


def train_stage_a_epoch(model: DrowningTemporalModel, batches: Iterable, optimizer: torch.optim.Optimizer,
                        device: str = "cpu", class_weights: torch.Tensor | None = None) -> dict[str, float | int]:
    """Train image head/encoder on image samples; caller must run this on Kaggle."""
    model.train()
    total_loss, sample_count = 0.0, 0
    for images, targets in batches:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = image_loss(model.forward_images(images), targets, class_weights)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
        sample_count += len(targets)
    if sample_count == 0:
        raise ValueError("image training loader produced no samples")
    return {"mean_loss": total_loss / sample_count, "samples_seen": sample_count}


def train_stage_b_epoch(model: DrowningTemporalModel, batches: Iterable, optimizer: torch.optim.Optimizer,
                        device: str = "cpu", class_weights: torch.Tensor | None = None,
                        image_aux_weight: float = 0.2) -> dict[str, float | int]:
    """Train causal temporal head with an image auxiliary loss to retain Stage A."""
    if image_aux_weight < 0:
        raise ValueError("image_aux_weight must be nonnegative")
    model.train()
    total_loss, sequence_count = 0.0, 0
    for frames, labels, valid_mask in batches:
        frames, labels, valid_mask = frames.to(device), labels.to(device), valid_mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(frames, valid_mask)
        last_labels = labels[:, -1] if labels.ndim == 2 else labels
        temporal_targets = labels if labels.ndim == 2 else labels[:, None].expand(-1, frames.shape[1])
        supervision_mask = valid_mask.bool()
        if labels.ndim == 2:
            loss_temporal = temporal_loss(output["temporal_logits"], temporal_targets, supervision_mask, class_weights)
        else:
            loss_temporal = image_loss(output["temporal_logits"][:, -1], last_labels, class_weights)
        loss_image = image_loss(output["image_logits"][:, -1], last_labels, class_weights)
        loss = loss_temporal + image_aux_weight * loss_image
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(frames)
        sequence_count += len(frames)
    if sequence_count == 0:
        raise ValueError("temporal training loader produced no sequences")
    return {"mean_loss": total_loss / sequence_count, "sequences_seen": sequence_count}


def classification_metrics(targets: Iterable[int], predictions: Iterable[int]) -> dict[str, object]:
    target_list, prediction_list = list(targets), list(predictions)
    if len(target_list) != len(prediction_list) or not target_list:
        raise ValueError("targets and predictions must have equal nonzero length")
    confusion = [[0 for _ in CLASS_KEYS] for _ in CLASS_KEYS]
    for actual, predicted in zip(target_list, prediction_list):
        if actual not in range(len(CLASS_KEYS)) or predicted not in range(len(CLASS_KEYS)):
            raise ValueError("class index outside ontology range 0..2")
        confusion[actual][predicted] += 1
    recalls, f1s = [], []
    for class_id in range(len(CLASS_KEYS)):
        tp = confusion[class_id][class_id]
        support = sum(confusion[class_id])
        predicted_count = sum(row[class_id] for row in confusion)
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {
        "sample_count": len(target_list),
        "accuracy": sum(confusion[i][i] for i in range(len(CLASS_KEYS))) / len(target_list),
        "macro_f1": sum(f1s) / len(CLASS_KEYS),
        "per_class_recall": dict(zip(CLASS_KEYS, recalls)),
        "confusion_matrix": confusion,
    }


@torch.inference_mode()
def evaluate_image_model(model: DrowningTemporalModel, batches: Iterable, *, device: str = "cpu") -> dict[str, object]:
    model.eval()
    targets, predictions = [], []
    for images, labels in batches:
        logits = model.forward_images(images.to(device))
        targets.extend(labels.tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return classification_metrics(targets, predictions)


@torch.inference_mode()
def evaluate_temporal_model(model: DrowningTemporalModel, batches: Iterable, *, device: str = "cpu") -> dict[str, object]:
    model.eval()
    targets, predictions = [], []
    for frames, labels, valid_mask in batches:
        output = model(frames.to(device), valid_mask.to(device))
        anchors = labels[:, -1]
        predictions.extend(output["temporal_logits"][:, -1].argmax(dim=-1).cpu().tolist())
        targets.extend(anchors.tolist())
    return classification_metrics(targets, predictions)
