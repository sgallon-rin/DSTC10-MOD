"""
python baselines/MemeBert-C/train_eval.py \
  --do_train --do_eval \
  --train_path data/MOD_full/ft_local/MOD-Dataset/validation/validation.json \
  --output_dir runs/memebert-c-test

python baselines/MemeBert-C/train_eval.py --do_eval --do_test 

python baselines/MemeBert-C/train_eval.py \
  --do_test \
  --checkpoint_path runs/memebert-c-test/memebert_c.pt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, AutoTokenizer
from transformers.utils import logging as hf_logging

from dataset import MemeDataset, get_data, tokenize
from model import MemeBERT

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Recall at position k in n candiates R_n@K 
# measure if the positive meme is ranked in the topK position of n candidates 
def recall_compute(candidates, targets, k=5): 
    if len(candidates) > k:
        candidates = candidates[:k]
    if targets not in candidates:
        return 0 
    else: 
        return 1 

# MAP: mean average precision 
def map_compute(candidates, targets, k=5): 
    if len(candidates) > k:
        candidates = candidates[:k]
    if targets not in candidates:
        return 0 
    else: 
        idx = candidates.index(targets) 
        return 1.0 / (1.0 + idx) 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MemeBERT-C training + validation/testing helper.")
    parser.add_argument("--bert_path", type=str, default="bert-base-chinese", help="Pretrained BERT path or HF id.")
    parser.add_argument("--train_path", type=str, default="data/MOD_full/ft_local/MOD-Dataset/train/c_train.json", help="Training JSON file.")
    parser.add_argument("--val_path", type=str, default="data/MOD_full/ft_local/MOD-Dataset/validation/val_task2.json", help="Validation JSON file (Task2 format).")
    parser.add_argument("--test_path", type=str, default="data/MOD-fix/MOD-test-fixed/c_test_hard/annotated_c_test_hard_task2_fixed.json", help="Test JSON file (Task2 format).")
    parser.add_argument("--output_dir", type=str, default="outputs/memebert_c", help="Directory for checkpoints/preds.")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Checkpoint to load instead of training.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1, help="Keep 1 to mimic original baseline.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=5000)
    parser.add_argument("--recall_ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--map_ks", type=int, nargs="+", default=[10])
    parser.add_argument("--topk_save", type=int, default=10, help="Top-K candidates to dump per sample.")
    parser.add_argument("--do_train", action="store_true", help="Run training loop before eval/test.")
    parser.add_argument("--do_eval", action="store_true", help="Run validation scoring.")
    parser.add_argument("--do_test", action="store_true", help="Run test inference.")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument("--prediction_prefix", type=str, default="pred", help="Prefix for saved prediction files.")
    return parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def acc_compute(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[bool, bool, bool]:
    _, idx = torch.sort(logits.squeeze(0))
    idx = idx.tolist()
    label = labels.item()
    return label in idx[-150:], label in idx[-90:], label in idx[-31:]


def train_one_epoch(
    model: MemeBERT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_steps: int,
    print_freq: int,
    epoch: int,
):
    model.train()
    avg_loss = AverageMeter()
    avg_acc1 = AverageMeter()
    avg_acc3 = AverageMeter()
    avg_acc5 = AverageMeter()
    optimizer.zero_grad()

    for iteration, batch in enumerate(loader, start=1):
        history_txt, labels = batch
        history_txt = history_txt.to(device)
        labels = labels.squeeze(0).to(device)

        loss, logits = model(input_ids=history_txt, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if iteration % grad_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss.update(loss.item())
        flag_5, flag_3, flag_1 = acc_compute(logits, labels)
        avg_acc5.update(int(flag_5))
        avg_acc3.update(int(flag_3))
        avg_acc1.update(int(flag_1))

        if print_freq > 0 and iteration % print_freq == 0:
            print(
                'Epoch:[{0}][{1}/{2}]\tLoss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Classify Acc {acc1.val:.3f} ({acc1.avg:.3f})|{acc3.val:.3f} ({acc3.avg:.3f})|'
                '{acc5.val:.3f} ({acc5.avg:.3f})'.format(
                    epoch, iteration, len(loader), loss=avg_loss, acc1=avg_acc1, acc3=avg_acc3, acc5=avg_acc5
                )
            )


def prepare_eval_tensor(sample: Dict, tokenizer) -> Optional[torch.Tensor]:
    history = sample.get("history", [])
    if not history:
        return None
    tokens: List[int] = []
    for turn in history:
        if "txt" not in turn:
            continue
        tokens += tokenize(turn["txt"], tokenizer)
    if not tokens:
        return None
    if len(tokens) >= 505:
        tokens = tokens[-500:]
    cls_token, sep_token = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
    input_ids = [cls_token] + tokens + [sep_token]
    return torch.LongTensor(input_ids).unsqueeze(0)


def gather_candidates(sample: Dict, num_labels: int) -> List[Tuple[int, str, bool]]:
    candidate = sample.get("candidate", {})
    cand_set = candidate.get("set", [])
    pairs = []
    if cand_set:
        for cid in cand_set:
            try:
                cid_int = int(cid)
            except ValueError:
                continue
            in_vocab = 0 <= cid_int < num_labels
            pairs.append((cid_int, str(cid), in_vocab))
    else:
        pairs = [(idx, str(idx).zfill(3), True) for idx in range(num_labels)]
    return pairs


def evaluate_split(
    model: MemeBERT,
    tokenizer,
    data_path: str,
    device: torch.device,
    recall_ks: Sequence[int],
    map_ks: Sequence[int],
    topk_to_save: int,
    output_file: Optional[Path] = None,
) -> Dict[str, float]:
    with open(data_path, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    metrics = {f"recall@{k}": 0.0 for k in recall_ks}
    metrics.update({f"map@{k}": 0.0 for k in map_ks})
    counted = 0
    predictions = []

    model.eval()
    with torch.no_grad():
        for idx, sample in enumerate(data):
            input_tensor = prepare_eval_tensor(sample, tokenizer)
            if input_tensor is None:
                continue
            input_tensor = input_tensor.to(device)
            outputs = model(input_ids=input_tensor)
            logits = outputs[1] if len(outputs) > 1 else outputs[0]
            logits = logits.squeeze(0).detach().cpu()

            candidate_pairs = gather_candidates(sample, model.num_labels)
            scored = []
            for cid_int, cid_str, in_vocab in candidate_pairs:
                score = logits[cid_int].item() if in_vocab else float("-inf")
                scored.append((score, cid_int, cid_str))

            ranked = sorted(scored, key=lambda x: x[0], reverse=True)
            ranked_ids = [item[1] for item in ranked]
            ranked_str = [item[2] for item in ranked]

            predictions.append(
                {
                    "index": sample.get("id", idx),
                    "top_candidates": ranked_str[:topk_to_save],
                }
            )

            if "answer" in sample and "img_id" in sample["answer"]:
                try:
                    target = str(sample["answer"]["img_id"])
                except ValueError:
                    continue
                counted += 1
                for k in recall_ks:
                    metrics[f"recall@{k}"] += recall_compute(ranked_str, target, k)
                for k in map_ks:
                    metrics[f"map@{k}"] += map_compute(ranked_str, target, k)

    if counted > 0:
        for key in metrics:
            metrics[key] /= counted
    else:
        metrics = {}

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(predictions, fout, ensure_ascii=False, indent=2)

    return metrics


def maybe_load_checkpoint(model: MemeBERT, ckpt_path: str, device: torch.device):
    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {ckpt_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    model = MemeBERT.from_pretrained(args.bert_path)
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.do_train:
        train_dialogs = get_data(tokenizer, args.train_path)
        train_dataset = MemeDataset(train_dialogs, tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
        )
        optimizer = AdamW(model.parameters(), lr=args.lr)
        for epoch in range(args.epochs):
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_steps=args.gradient_accumulation_steps,
                print_freq=args.print_freq,
                epoch=epoch,
            )
        ckpt_path = output_dir / "memebert_c.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
    else:
        maybe_load_checkpoint(model, args.checkpoint_path, device)

    if args.do_eval and args.val_path:
        val_out = output_dir / f"{args.prediction_prefix}_val.json"
        metrics = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            data_path=args.val_path,
            device=device,
            recall_ks=args.recall_ks,
            map_ks=args.map_ks,
            topk_to_save=args.topk_save,
            output_file=val_out,
        )
        print(f"Validation metrics: {metrics}")

    if args.do_test and args.test_path:
        test_out = output_dir / f"{args.prediction_prefix}_test.json"
        metrics = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            data_path=args.test_path,
            device=device,
            recall_ks=args.recall_ks,
            map_ks=args.map_ks,
            topk_to_save=args.topk_save,
            output_file=test_out,
        )
        if metrics:
            print(f"Test metrics: {metrics}")
        else:
            print(f"Wrote test predictions to {test_out}")


if __name__ == "__main__":
    set_seed(42)
    main()
