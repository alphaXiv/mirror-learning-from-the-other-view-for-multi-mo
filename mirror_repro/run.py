from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForImageTextToText, AutoProcessor

from .data import Problem, load_split, prepare_geometry3k, renderer_audit, shared_data_root, view_payload


VIEWS = ("text", "image", "combined")
STUDENT_VIEWS = ("text", "image")


def setup_distributed() -> tuple[int, int, int, torch.device]:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def seed_everything(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def move(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def messages_for(text: str, has_image: bool, answer: str | None = None) -> list[dict[str, Any]]:
    if has_image:
        content: Any = [{"type": "image"}, {"type": "text", "text": text}]
    else:
        content = text
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": f"Final answer: {answer}"})
    return messages


def encode(
    processor: Any,
    problem: Problem,
    view: str,
    device: torch.device,
    max_side: int,
    answer: str | None = None,
) -> tuple[dict[str, torch.Tensor], int]:
    text, image = view_payload(problem, view, max_side)
    prompt_messages = messages_for(text, image is not None)
    prompt = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_inputs = processor(
        text=[prompt], images=[image] if image is not None else None, return_tensors="pt"
    )
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    if answer is None:
        return move(prompt_inputs, device), prompt_len
    full = processor.apply_chat_template(
        messages_for(text, image is not None, answer), tokenize=False, add_generation_prompt=False
    )
    full_inputs = processor(
        text=[full], images=[image] if image is not None else None, return_tensors="pt"
    )
    return move(full_inputs, device), prompt_len


def extract_answer(text: str) -> str:
    hits = re.findall(r"final\s*answer\s*[:：]\s*\(?\s*([ABCD])\b", text, flags=re.I)
    if hits:
        return hits[-1].upper()
    lone = re.findall(r"(?:^|\s)\(?([ABCD])\)?(?:\s|$)", text.strip(), flags=re.I)
    return lone[-1].upper() if lone else "?"


@torch.inference_mode()
def predict(
    model: Any,
    processor: Any,
    problem: Problem,
    view: str,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[str, str]:
    inputs, prompt_len = encode(
        processor, problem, view, device, int(cfg["image_max_side"])
    )
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=int(cfg["max_new_tokens"]),
        use_cache=True,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    decoded = processor.batch_decode(
        output[:, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return extract_answer(decoded), decoded[-240:]


def gather_objects(local: Any, world: int) -> list[Any]:
    gathered: list[Any] = [None for _ in range(world)]
    dist.all_gather_object(gathered, local)
    return gathered


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row["id"]))
    n = len(rows)
    accuracy = {
        view: sum(row["pred"][view] == row["gold"] for row in rows) / max(1, n)
        for view in VIEWS
    }
    all_agree = sum(len(set(row["pred"].values())) == 1 for row in rows) / max(1, n)
    all_correct = sum(all(row["pred"][v] == row["gold"] for v in VIEWS) for row in rows) / max(1, n)
    any_correct = sum(any(row["pred"][v] == row["gold"] for v in VIEWS) for row in rows) / max(1, n)
    complementary = sum(
        any(row["pred"][v] == row["gold"] for v in VIEWS)
        and not all(row["pred"][v] == row["gold"] for v in VIEWS)
        for row in rows
    ) / max(1, n)
    return {
        "n": n,
        "accuracy": accuracy,
        "mean_view_accuracy": sum(accuracy.values()) / len(VIEWS),
        "all_view_agreement": all_agree,
        "all_view_correct": all_correct,
        "oracle_any_view_correct": any_correct,
        "complementary_success": complementary,
        "disagreement": 1.0 - all_agree,
        "predictions": rows,
    }


def evaluate(
    model: Any,
    processor: Any,
    problems: list[Problem],
    cfg: dict[str, Any],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    local: list[dict[str, Any]] = []
    for problem in problems[rank::world]:
        predictions: dict[str, str] = {}
        snippets: dict[str, str] = {}
        for view in VIEWS:
            pred, snippet = predict(model, processor, problem, view, cfg, device)
            predictions[view] = pred
            snippets[view] = snippet
        local.append(
            {"id": problem.problem_id, "gold": problem.answer, "pred": predictions, "sample": snippets}
        )
    flat = [row for shard in gather_objects(local, world) for row in shard]
    return summarize_predictions(flat)


def teacher_scan(
    model: Any,
    processor: Any,
    problems: list[Problem],
    cfg: dict[str, Any],
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[dict[str, str], dict[str, Any]]:
    method = str(cfg["method"])
    local_map: dict[str, str] = {}
    local_stats: list[dict[str, Any]] = []
    for problem in problems[rank::world]:
        if method == "fixed_combined":
            predictions = {"combined": predict(model, processor, problem, "combined", cfg, device)[0]}
            selected = "combined"
        elif method == "fixed_text":
            predictions = {"text": predict(model, processor, problem, "text", cfg, device)[0]}
            selected = "text"
        else:
            predictions = {
                view: predict(model, processor, problem, view, cfg, device)[0] for view in VIEWS
            }
            rewards = {view: int(predictions[view] == problem.answer) for view in VIEWS}
            best = max(rewards.values())
            tied = [view for view in VIEWS if rewards[view] == best]
            selected = tied[(int(problem.problem_id) + int(cfg["seed"])) % len(tied)]
        local_map[problem.problem_id] = selected
        local_stats.append(
            {
                "id": problem.problem_id,
                "selected": selected,
                "pred": predictions,
                "gold": problem.answer,
            }
        )
    map_parts = gather_objects(local_map, world)
    stat_parts = gather_objects(local_stats, world)
    teacher_map = {key: value for part in map_parts for key, value in part.items()}
    stats = [row for part in stat_parts for row in part]
    counts = Counter(row["selected"] for row in stats)
    asymmetry = 0
    for row in stats:
        if len(row["pred"]) == 3:
            correctness = [row["pred"][v] == row["gold"] for v in VIEWS]
            asymmetry += int(any(correctness) and not all(correctness))
    return teacher_map, {
        "selection_counts": dict(counts),
        "selection_fractions": {key: value / max(1, len(stats)) for key, value in counts.items()},
        "train_asymmetry_fraction": asymmetry / max(1, len(stats)),
    }


def answer_logits(
    raw_model: Any,
    processor: Any,
    problem: Problem,
    view: str,
    cfg: dict[str, Any],
    device: torch.device,
    disable_adapter: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, prompt_len = encode(
        processor,
        problem,
        view,
        device,
        int(cfg["image_max_side"]),
        answer=problem.answer,
    )
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    model_inputs = dict(inputs)
    model_inputs["labels"] = labels
    adapter_context = raw_model.disable_adapter() if disable_adapter else contextlib.nullcontext()
    with adapter_context:
        output = raw_model(**model_inputs, use_cache=False)
    shifted = output.logits[:, :-1, :]
    mask = labels[:, 1:] != -100
    return output.loss, shifted[mask]


def train_lora(
    model: Any,
    processor: Any,
    problems: list[Problem],
    teacher_map: dict[str, str],
    cfg: dict[str, Any],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    lora = LoraConfig(
        r=int(cfg["lora_rank"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()
    ddp = DDP(model, device_ids=[device.index], find_unused_parameters=True)
    raw_model = ddp.module
    optimizer = torch.optim.AdamW(
        (p for p in raw_model.parameters() if p.requires_grad),
        lr=float(cfg["learning_rate"]),
        weight_decay=0.01,
    )
    samples = [(problem, view) for problem in problems for view in STUDENT_VIEWS]
    rng = random.Random(int(cfg["seed"]))
    rng.shuffle(samples)
    local_samples = samples[rank::world]
    accum = int(cfg["gradient_accumulation"])
    coefficient = float(cfg["reverse_kl_coefficient"])
    use_rkl = str(cfg["method"]) in {"mirror", "fixed_combined", "fixed_text"}
    trajectory: list[dict[str, float | int]] = []
    global_micro = 0
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg["epochs"])):
        for local_index, (problem, student_view) in enumerate(local_samples):
            student_loss, student_logits = answer_logits(
                raw_model, processor, problem, student_view, cfg, device, disable_adapter=False
            )
            rkl = torch.zeros((), device=device)
            teacher_view = teacher_map.get(problem.problem_id, student_view)
            if use_rkl and teacher_view != student_view:
                with torch.no_grad():
                    _, teacher_logits = answer_logits(
                        raw_model,
                        processor,
                        problem,
                        teacher_view,
                        cfg,
                        device,
                        disable_adapter=True,
                    )
                token_count = min(student_logits.shape[0], teacher_logits.shape[0])
                s = student_logits[-token_count:].float()
                t = teacher_logits[-token_count:].float()
                log_s = F.log_softmax(s, dim=-1)
                log_t = F.log_softmax(t, dim=-1)
                rkl = (log_s.exp() * (log_s - log_t)).sum(dim=-1).mean()
            loss = (student_loss + coefficient * rkl) / accum
            loss.backward()
            global_micro += 1
            if global_micro % accum == 0 or local_index == len(local_samples) - 1:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                values = torch.tensor(
                    [float(student_loss.detach()), float(rkl.detach())],
                    device=device,
                )
                dist.all_reduce(values, op=dist.ReduceOp.SUM)
                values /= world
                if rank == 0:
                    point = {
                        "step": optimizer_steps,
                        "supervised_loss": float(values[0]),
                        "reverse_kl": float(values[1]),
                    }
                    trajectory.append(point)
                    print("TRAIN_STEP " + json.dumps(point, sort_keys=True), flush=True)
    return {
        "model": ddp.module,
        "optimizer_steps": optimizer_steps,
        "micro_steps_per_rank": global_micro,
        "trajectory": trajectory,
    }


def strip_samples(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = dict(metrics)
    for row in compact.get("predictions", []):
        row.pop("sample", None)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    started = time.time()
    rank, world, local_rank, device = setup_distributed()
    seed_everything(int(cfg["seed"]), rank)
    if rank == 0:
        print("CONFIG_JSON=" + json.dumps(cfg, sort_keys=True), flush=True)
        print(f"DISTRIBUTED world_size={world} local_rank={local_rank}", flush=True)
        prepare_geometry3k(shared_data_root())
    dist.barrier()
    roots = {"train": shared_data_root() / "train", "val": shared_data_root() / "val"}
    train_problems = load_split(roots["train"], int(cfg["train_problems"]), int(cfg["seed"]))
    eval_problems = load_split(roots["val"], int(cfg["eval_problems"]), int(cfg["seed"]) + 991)
    if rank == 0:
        print("DATASET_AUDIT=" + json.dumps(renderer_audit(eval_problems), sort_keys=True), flush=True)
    processor = AutoProcessor.from_pretrained(cfg["model"])
    model = AutoModelForImageTextToText.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = True
    before = evaluate(model, processor, eval_problems, cfg, rank, world, device)
    if rank == 0:
        print("BASELINE_METRICS=" + json.dumps(strip_samples(before), sort_keys=True), flush=True)
    teacher_info: dict[str, Any] = {}
    training_info: dict[str, Any] = {}
    after = before
    if cfg["method"] != "baseline":
        if cfg["method"] == "sft":
            teacher_map = {problem.problem_id: "text" for problem in train_problems}
            teacher_info = {"selection": "not used"}
        else:
            teacher_map, teacher_info = teacher_scan(
                model, processor, train_problems, cfg, rank, world, device
            )
        trained = train_lora(
            model, processor, train_problems, teacher_map, cfg, rank, world, device
        )
        model = trained.pop("model")
        training_info = trained
        model.config.use_cache = True
        after = evaluate(model, processor, eval_problems, cfg, rank, world, device)
    if rank == 0:
        before_rows = {row["id"]: row for row in before["predictions"]}
        after_rows = {row["id"]: row for row in after["predictions"]}
        initially_disagree = [
            key for key, row in before_rows.items() if len(set(row["pred"].values())) > 1
        ]
        resolved_all_correct = sum(
            all(after_rows[key]["pred"][v] == after_rows[key]["gold"] for v in VIEWS)
            for key in initially_disagree
        )
        resolved_agreement = sum(
            len(set(after_rows[key]["pred"].values())) == 1 for key in initially_disagree
        )
        result = {
            "schema": 1,
            "paper_id": "2607.21552",
            "dataset": "MIT-licensed Geometry3K/InterGPS official train/validation splits",
            "method": cfg["method"],
            "seed": cfg["seed"],
            "model": cfg["model"],
            "gpu_model_required": "NVIDIA RTX PRO 6000 Blackwell",
            "gpu_count": world,
            "before": strip_samples(before),
            "after": strip_samples(after),
            "teacher": teacher_info,
            "training": training_info,
            "disagreement_resolution": {
                "initially_disagree_n": len(initially_disagree),
                "became_all_correct_n": resolved_all_correct,
                "became_agreement_n": resolved_agreement,
                "became_all_correct_fraction": resolved_all_correct / max(1, len(initially_disagree)),
                "became_agreement_fraction": resolved_agreement / max(1, len(initially_disagree)),
            },
            "elapsed_seconds": time.time() - started,
            "scope_note": (
                "Fresh Kubernetes evidence. Geometry3K substitutes for unavailable ODA-Data; "
                "LoRA answer supervision substitutes for full 16-rollout GRPO. MIRROR branches "
                "use a frozen-base, gold-trajectory token-level reverse-KL approximation."
            ),
        }
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
        print(
            f"RUN_COMPLETE method={cfg['method']} elapsed_seconds={result['elapsed_seconds']:.1f}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

