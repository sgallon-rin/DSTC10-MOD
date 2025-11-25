import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer

from dataset import MemeDataset, build_input_from_segments, get_data, tokenize
from model import MemeBERT
from score_script.task2_score import map_compute, recall_compute


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MemeBERT (Chinese) training script with validation and easy/hard test inference.",
    )
    parser.add_argument("--bert_path", type=str, default="google-bert/bert-base-chinese", help="Pretrained BERT path or huggingface id.")
    parser.add_argument("--train_path", type=str, default="data/MOD_full/ft_local/MOD-Dataset/train/c_train.json", help="Training JSON file.")
    parser.add_argument("--val_path", type=str, default="data/MOD_full/ft_local/MOD-Dataset/validation/val_task2.json", help="Validation JSON file (Task2 format).")
    parser.add_argument("--test_easy_path", type=str, default="data/MOD-fix/MOD-test-fixed/c_test_easy/annotated_c_test_easy_task2_fixed.json", help="Easy test JSON file (Task2 format).")
    parser.add_argument("--test_hard_path", type=str, default="data/MOD-fix/MOD-test-fixed/c_test_hard/annotated_c_test_hard_task2_fixed.json", help="Hard test JSON file (Task2 format).")
    parser.add_argument("--output_dir", type=str, default="outputs/memebert_c", help="Directory to save checkpoints/preds.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training.")
    parser.add_argument("--save_best", action="store_true", help="Save best checkpoint based on validation metric.")
    parser.add_argument("--selection_metric", type=str, default="recall@1", help="Metric key used for model selection.")
    parser.add_argument("--recall_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--map_ks", type=int, nargs="+", default=[10])
    parser.add_argument("--train_monitor_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--output_top_k",
        type=int,
        default=5,
        help="How many predictions to store per sample when dumping inference results.",
    )
    parser.add_argument("--prediction_prefix", type=str, default="predictions", help="Filename prefix for saved predictions.")
    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def build_collate_fn(pad_token_id: int):
    def collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, labels = zip(*batch)
        lengths = [seq.size(0) for seq in input_ids]
        max_len = max(lengths)
        padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        for idx, seq in enumerate(input_ids):
            padded[idx, : seq.size(0)] = seq
        labels_tensor = torch.stack(labels).squeeze(-1)
        return padded, labels_tensor

    return collate_fn


def encode_history(history: List[Dict], tokenizer) -> torch.Tensor:
    turns = copy.deepcopy(history)
    for turn in turns:
        if "txt" in turn:
            turn["txt"] = tokenize(turn["txt"], tokenizer)
    history_txt, _ = build_input_from_segments(turns, tokenizer)
    return torch.LongTensor(history_txt)


def rank_candidates(
    logits: torch.Tensor,
    candidate_set: Optional[Sequence],
    num_labels: int,
) -> Tuple[List[str], List[int]]:
    if candidate_set:
        candidates = [(str(cid), int(cid)) for cid in candidate_set]
    else:
        candidates = [(str(idx).zfill(3), idx) for idx in range(num_labels)]

    scored = sorted(candidates, key=lambda item: logits[item[1]].item(), reverse=True)
    ranked_str = [item[0] for item in scored]
    ranked_int = [item[1] for item in scored]
    return ranked_str, ranked_int


def evaluate_split(
    model: MemeBERT,
    tokenizer,
    data_path: str,
    device: torch.device,
    recall_ks: Sequence[int],
    map_ks: Sequence[int],
    top_k_to_save: int,
    split_name: str,
) -> Tuple[Dict[str, float], List[Dict]]:
    with open(data_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    has_answers = all("answer" in sample and "img_id" in sample["answer"] for sample in data)
    metrics = {f"recall@{k}": 0.0 for k in recall_ks}
    metrics.update({f"map@{k}": 0.0 for k in map_ks})
    predictions = []

    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(data):
            input_ids = encode_history(sample["history"], tokenizer).unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids)
            logits = outputs[1] if len(outputs) > 1 else outputs[0]
            logits = logits.squeeze(0).detach().cpu()

            candidate_set = None
            if "candidate" in sample and isinstance(sample["candidate"], dict):
                candidate_set = sample["candidate"].get("set")

            ranked_str, ranked_int = rank_candidates(logits, candidate_set, model.num_labels)

            predictions.append(
                {
                    "index": sample.get("id", idx),
                    "top_candidates": ranked_str[:top_k_to_save],
                    "split": split_name,
                }
            )

            if has_answers:
                target_id = int(sample["answer"]["img_id"])
                for k in recall_ks:
                    metrics[f"recall@{k}"] += recall_compute(ranked_int, target_id, k)
                for k in map_ks:
                    metrics[f"map@{k}"] += map_compute(ranked_int, target_id, k)

    if has_answers:
        total = len(data)
        for key in metrics:
            metrics[key] /= max(total, 1)
    else:
        metrics = {}
    return metrics, predictions


def compute_topk_hits(logits: torch.Tensor, labels: torch.Tensor, ks: Sequence[int]) -> Dict[int, float]:
    results = {}
    for k in ks:
        topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices
        matches = topk.eq(labels.unsqueeze(-1))
        results[k] = matches.any(dim=-1).float().mean().item()
    return results


def train_one_epoch(
    model: MemeBERT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_accum_steps: int,
    max_grad_norm: float,
    log_steps: int,
    monitor_ks: Sequence[int],
    epoch: int,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meters = {k: AverageMeter() for k in monitor_ks}

    optimizer.zero_grad()
    for step, batch in enumerate(loader, start=1):
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0] / grad_accum_steps
        logits = outputs[1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        if step % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_meter.update(loss.item() * grad_accum_steps, n=labels.size(0))

        with torch.no_grad():
            topk_hits = compute_topk_hits(logits.detach(), labels, monitor_ks)
        for k, hit in topk_hits.items():
            acc_meters[k].update(hit, labels.size(0))

        if log_steps and step % log_steps == 0:
            acc_str = " | ".join(f"Top{k}: {acc_meters[k].avg:.3f}" for k in monitor_ks)
            print(
                f"Epoch {epoch} Step {step}/{len(loader)} "
                f"Loss {loss_meter.avg:.4f} "
                f"{acc_str}"
            )

    stats = {"loss": loss_meter.avg}
    stats.update({f"top{k}": meter.avg for k, meter in acc_meters.items()})
    return stats


def save_predictions(predictions: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(predictions, fout, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    model = MemeBERT.from_pretrained(args.bert_path)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Loading training data ...")
    train_dialogs = get_data(tokenizer, args.train_path)
    train_dataset = MemeDataset(train_dialogs, tokenizer)
    collate_fn = build_collate_fn(tokenizer.pad_token_id or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=not args.no_cuda,
        collate_fn=collate_fn,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_metric_val = float("-inf")
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_accum_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_steps=args.log_steps,
            monitor_ks=args.train_monitor_ks,
            epoch=epoch,
        )
        print(f"Epoch {epoch} finished. Loss {train_stats['loss']:.4f}")

        if args.val_path:
            val_metrics, _ = evaluate_split(
                model=model,
                tokenizer=tokenizer,
                data_path=args.val_path,
                device=device,
                recall_ks=args.recall_ks,
                map_ks=args.map_ks,
                top_k_to_save=args.output_top_k,
                split_name="val",
            )
            print(f"Validation metrics: {val_metrics}")
            sel_metric_value = val_metrics.get(args.selection_metric, None)
            if args.save_best and sel_metric_value is not None and sel_metric_value > best_metric_val:
                best_metric_val = sel_metric_value
                best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    if args.save_best and best_state_dict is not None:
        ckpt_path = output_dir / "best_model.pt"
        torch.save(best_state_dict, ckpt_path)
        print(f"Best checkpoint saved to {ckpt_path}")
    else:
        ckpt_path = output_dir / "last_model.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Final checkpoint saved to {ckpt_path}")

    for split_name, data_path in [
        ("val", args.val_path),
        ("test_easy", args.test_easy_path),
        ("test_hard", args.test_hard_path),
    ]:
        if not data_path:
            continue
        metrics, predictions = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            data_path=data_path,
            device=device,
            recall_ks=args.recall_ks,
            map_ks=args.map_ks,
            top_k_to_save=args.output_top_k,
            split_name=split_name,
        )
        if metrics:
            print(f"{split_name} metrics: {metrics}")
        out_file = output_dir / f"{args.prediction_prefix}_{split_name}.json"
        save_predictions(predictions, out_file)
        print(f"Saved {split_name} predictions to {out_file}")


if __name__ == "__main__":
    main()
