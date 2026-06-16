#!/usr/bin/env python3
"""
Qwen3.5-35B-A3B (MoE) in-context learning evaluation on glaucoma fundus images.

Loads 6 in-context examples (image + 6-step CoT JSON) from
COT_Eyes/In context/In Context Training Dataset/, then evaluates the model
on the In Context Test Dataset (727 images), saving predictions and metrics.

Inspired by the Qwen-VL inference recipe in SHBI-Final-Project/src/{models,prompts,inference}.py
(yuzhench@100.109.238.5:/home/yuzhench/Desktop/SHBI-Final-Project/).

Usage:
    # Smoke test (10 samples)
    python run_qwen35_incontext.py --num-samples 10

    # Full eval, default model (Qwen/Qwen3.5-35B-A3B), bf16 on a single big GPU
    python run_qwen35_incontext.py

    # 4-bit if VRAM is tight (e.g. single 48GB A6000)
    python run_qwen35_incontext.py --load-in-4bit

    # Override model
    python run_qwen35_incontext.py --model-name Qwen/Qwen3-VL-30B-A3B-Instruct
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (edit if running on a different machine layout)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
DEFAULT_ICL_DIR = _HERE / "DATA/in_context_examples"
DEFAULT_TEST_PHOTOS = _HERE / "DATA/test_120/photos"
DEFAULT_TEST_DESCRIPTIONS = _HERE / "DATA/test_120/descriptions"
DEFAULT_OUTPUT_DIR = _HERE / "outputs"

LABEL_VOCAB = {"not likely", "borderline", "likely"}

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
# ---- Prompt strings — kept BYTE-IDENTICAL to Grace's `few-shot qwen.ipynb`
# so this experiment is directly comparable to her Qwen2.5-VL-7B in-context numbers
# (59.3% acc / 0.339 macro F1).  Differences from her notebook are explicitly
# called out in the README.
SYSTEM_PROMPT = (
    "You are an expert ophthalmologist interpreting a retinal fundus photo for "
    "glaucoma,all images are of sufficient quality for diagnosis."
)

# ICL examples use the JSON's own "input" field, which is:
#   "Analyze the optic nerve head and generate a clinical reasoning chain to
#    assess for possible glaucoma."
# (loaded per-example below; do NOT hard-code it here)

# Final test-query text — Grace's notebook adds the explicit class list to the
# final query (but not to the ICL examples).
TEST_QUERY_TEXT = (
    "Analyze the optic nerve head and generate a clinical reasoning chain to "
    "assess for possible glaucoma. Give your final diagnosis in one of the "
    "following categories: not likely / borderline / likely."
)


def _format_example_assistant_text(parsed_json: dict) -> str:
    """Match Grace exactly: pass the raw dict through str().
    Grace's notebook does {"type": "text", "text": cot_output} where cot_output
    is the dict loaded from JSON; the chat template stringifies it as Python repr.
    We do the same here so the model sees identical input."""
    return str(parsed_json["output"])


def load_icl_examples(icl_dir: Path) -> list[dict]:
    """Load every (image, json) pair from the In Context Training Dataset.
    Mirrors Grace's loop: uses the JSON's own `input` text for the user turn
    and the dict-stringified `output` for the assistant turn."""
    examples = []
    for json_path in sorted(icl_dir.glob("*.json")):
        stem = json_path.stem
        img_path = icl_dir / f"{stem}.jpg"
        if not img_path.exists():
            logging.warning("Missing image for %s — skipping", stem)
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        examples.append({
            "id": stem,
            "image_path": str(img_path),
            "user_text": data["input"],                       # ← Grace uses cot_input
            "assistant_text": _format_example_assistant_text(data),  # str(dict)
            "diagnosis": data["output"]["Diagnosis Classification"],
        })
    return examples


def load_test_records(photos_dir: Path, descriptions_dir: Path) -> list[dict]:
    """Pair each test image with its ground-truth json (when present)."""
    records = []
    for img_path in sorted(photos_dir.glob("*.jpg")):
        stem = img_path.stem
        desc_path = descriptions_dir / f"{stem}.json"
        true_label = None
        reference_dict = None
        if desc_path.exists():
            try:
                d = json.loads(desc_path.read_text(encoding="utf-8"))
                true_label = d["output"].get("Diagnosis Classification")
                reference_dict = d
            except Exception as e:
                logging.warning("Failed to parse %s: %s", desc_path, e)
        records.append({
            "id": stem,
            "image_path": str(img_path),
            "true_diagnosis": (true_label or "").strip().lower(),
            "reference": reference_dict,
        })
    return records


def build_messages(icl_examples: list[dict], test_image_path: str) -> list[dict]:
    """Build a Qwen chat-format messages list with N few-shot examples
    followed by the test image + question.

    Structure mirrors Grace's `few-shot qwen.ipynb`:
    - System prompt (the short "expert ophthalmologist" string)
    - For each ICL example:
        user:      [{"type":"image","image":...}, {"type":"text","text":<JSON's `input`>}]
        assistant: [{"type":"text","text":<str(JSON's `output` dict)>}]
    - Final user turn:
        user:      [{"type":"image","image":...}, {"type":"text","text":TEST_QUERY_TEXT}]
    """
    messages: list[dict] = [
        {"role": "system",
         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]
    for ex in icl_examples:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": ex["image_path"]},
                {"type": "text", "text": ex["user_text"]},
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": ex["assistant_text"]}],
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": test_image_path},
            {"type": "text", "text": TEST_QUERY_TEXT},
        ],
    })
    return messages


# ---------------------------------------------------------------------------
# Model loading (mirrors SHBI-Final-Project/src/models.py)
# ---------------------------------------------------------------------------
def _resolve_model_class(name: str):
    lower = name.lower()
    try:
        if "qwen3.5" in lower and re.search(r"a\d+b", lower):
            from transformers import Qwen3_5MoeForConditionalGeneration as cls
            return cls
        if "qwen3.5" in lower:
            from transformers import Qwen3_5ForConditionalGeneration as cls
            return cls
        if "qwen3-vl" in lower and re.search(r"a\d+b", lower):
            from transformers import Qwen3VLMoeForConditionalGeneration as cls
            return cls
        if "qwen3-vl" in lower:
            from transformers import Qwen3VLForConditionalGeneration as cls
            return cls
        if "qwen2.5-vl" in lower:
            from transformers import Qwen2_5_VLForConditionalGeneration as cls
            return cls
    except ImportError as e:
        logging.warning("Specific Qwen-VL class unavailable (%s); falling back.", e)
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText


def load_model(name: str, dtype: str = "bfloat16",
               load_in_4bit: bool = False, load_in_8bit: bool = False,
               device_map: str = "auto",
               attn_implementation: str = "sdpa"):
    from transformers import AutoProcessor

    torch_dtype = {"bfloat16": torch.bfloat16,
                   "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    cls = _resolve_model_class(name)
    logging.info("Loading %s via %s (4bit=%s 8bit=%s)",
                 name, cls.__name__, load_in_4bit, load_in_8bit)
    model = cls.from_pretrained(
        name,
        torch_dtype=torch_dtype if quant_config is None else None,
        quantization_config=quant_config,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    processor = AutoProcessor.from_pretrained(name)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    return model, processor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_one(model, processor, messages: list[dict], max_new_tokens: int = 2048,
                  greedy: bool = False) -> str:
    """Run a single (multi-turn) message through the model. Returns decoded
    new tokens only (assistant reply).

    greedy=True -> deterministic argmax decoding (do_sample=False). Needed for
    long text-only prompts where temperature=0.001 sampling overflows softmax
    (torch.multinomial: 'probability tensor contains inf/nan'). Default False
    keeps the original image-ICL runs byte-identical."""
    from qwen_vl_utils import process_vision_info

    # Qwen3.5 has a "thinking mode" — disable to avoid <think>...</think> wrappers.
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except TypeError:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        pad = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        if greedy:
            # Deterministic argmax. No temperature/softmax/multinomial -> cannot
            # hit the 'probability tensor contains inf/nan' assert that
            # temperature=0.001 sampling triggers on these prompts.
            gen_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=pad,
            )
        else:
            # Match Grace's vLLM SamplingParams(temperature=0.001) — near-greedy
            # but technically sampling; stays bit-aligned with her notebook.
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.001,
                top_p=1.0,
                top_k=0,
                num_beams=1,
                pad_token_id=pad,
            )
    trimmed = gen_ids[0][len(inputs.input_ids[0]):]
    return processor.tokenizer.decode(trimmed, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Diagnosis parsing (label extraction)
# ---------------------------------------------------------------------------
NO_DIAG_SENTINEL = "NO_DIAGNOSIS_FOUND"


def parse_diagnosis(text: str) -> tuple[str, str]:
    """Extract the diagnosis label from generated text.

    Returns (label, status):
      label:
        - 'not likely' / 'borderline' / 'likely' on success
        - 'NO_DIAGNOSIS_FOUND' if no label could be extracted
      status:
        - 'ok_anchor'    — found via "Diagnosis Classification" anchor (clean)
        - 'ok_fallback'  — no anchor, found a label via keyword scan (less reliable)
        - 'no_diagnosis_found' — neither anchor nor keyword scan worked

    Handles formats:
      - dict-repr style:  "'Diagnosis Classification': 'Likely'"  (Grace-style ICL)
      - free-text style:  "Diagnosis Classification: Likely"
      - markdown style:   "**Diagnosis Classification**: Likely"
      - bare label:       just "Likely" somewhere in the text
    """
    if not text or not text.strip():
        return NO_DIAG_SENTINEL, "no_diagnosis_found"
    lower = text.lower()
    # 1) Look for the LAST occurrence of "diagnosis classification" anchor —
    #    model may echo ICL examples earlier in its output, so we want the final one.
    anchor = lower.rfind("diagnosis classification")
    if anchor != -1:
        window = lower[anchor : anchor + 80]
        for kw in ["not likely", "borderline", "likely"]:
            if kw in window:
                return kw, "ok_anchor"
    # 2) No anchor (or anchor present but no label nearby) — fallback: scan whole text.
    for kw in ["not likely", "borderline", "likely"]:
        if kw in lower:
            return kw, "ok_fallback"
    # 3) Total failure.
    return NO_DIAG_SENTINEL, "no_diagnosis_found"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_vllm(name: str, tensor_parallel_size: int = 2,
              dtype: str = "bfloat16", max_images: int = 8,
              max_model_len: int | None = None):
    """vLLM engine (tensor-parallel + paged KV). SAME AutoProcessor as the
    HF path so the chat template / image preprocessing — hence the prompt —
    is identical and the two backends are directly comparable.

    NOTE: needs `pip install vllm` matched to the installed torch, and the
    vLLM build must support this multimodal MoE arch. Smoke before trusting.
    limit_mm_per_prompt['image'] must cover the 6 ICL imgs + 1 query (>=7).
    """
    from vllm import LLM
    from transformers import AutoProcessor
    llm = LLM(model=name, tensor_parallel_size=tensor_parallel_size,
              dtype=dtype, trust_remote_code=True,
              limit_mm_per_prompt={"image": max_images},
              max_model_len=max_model_len,
              gpu_memory_utilization=0.90)
    processor = AutoProcessor.from_pretrained(name)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
    return llm, processor


def vllm_generate(llm, processor, messages: list[dict],
                  max_new_tokens: int = 2048, greedy: bool = False) -> str:
    """One sample via vLLM. Builds the SAME prompt text as generate_one
    (processor chat template) and the SAME image tensors (qwen_vl_utils),
    passed as multi_modal_data — so output is comparable to the hf path."""
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    mm: dict = {}
    if image_inputs:
        mm["image"] = image_inputs
    if video_inputs:
        mm["video"] = video_inputs
    req = {"prompt": prompt}
    if mm:
        req["multi_modal_data"] = mm
    sp = SamplingParams(
        temperature=0.0 if greedy else 0.001,
        max_tokens=max_new_tokens,
    )
    out = llm.generate(req, sp)
    return out[0].outputs[0].text


def vllm_generate_batch(llm, processor, messages_list: list[list[dict]],
                        max_new_tokens: int = 2048,
                        greedy: bool = False) -> list[str]:
    """Submit ALL prompts in ONE llm.generate call so vLLM's continuous
    batching runs them concurrently (this is where the speedup is — a
    per-sample loop would defeat vLLM entirely). Returns texts aligned to
    messages_list order."""
    from vllm import SamplingParams
    from qwen_vl_utils import process_vision_info
    reqs = []
    for messages in messages_list:
        prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        mm: dict = {}
        if image_inputs:
            mm["image"] = image_inputs
        if video_inputs:
            mm["video"] = video_inputs
        r = {"prompt": prompt}
        if mm:
            r["multi_modal_data"] = mm
        reqs.append(r)
    sp = SamplingParams(temperature=0.0 if greedy else 0.001,
                        max_tokens=max_new_tokens)
    outs = llm.generate(reqs, sp)            # vLLM continuous-batches these
    return [o.outputs[0].text for o in outs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--device-map", default="cuda:0",
                    help="Single-GPU by default (cuda:0). Use 'auto' if the model "
                         "doesn't fit on one card and you want tensor parallel. "
                         "Qwen3.5-35B-A3B in bf16 is ~70 GB and SHOULD fit on a "
                         "single 97 GB RTX PRO 6000 with room for KV cache.")
    ap.add_argument("--greedy", action="store_true",
                    help="Deterministic argmax decoding (do_sample=False). "
                         "REQUIRED with --device-map auto / multi-GPU: "
                         "temperature sampling on a sharded model produces "
                         "garbage (multinomial inf/nan). Single-GPU may omit.")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"],
                    help="Inference engine. 'hf' (default) = the original "
                         "transformers model.generate path (slower, batch=1, "
                         "device_map pipeline-shard) — UNCHANGED, always works. "
                         "'vllm' = vLLM engine (tensor-parallel + continuous "
                         "batching, much faster) — requires `pip install vllm` "
                         "(version-matched to torch) and a smoke first.")
    ap.add_argument("--tensor-parallel-size", type=int, default=2,
                    help="vLLM only: #GPUs to tensor-parallel the model over "
                         "(2 = Qwen3.5-35B bf16 across 2x48GB).")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Data-parallel split of the test set across N "
                         "independent replicas (run each on its own GPU "
                         "pair). 1 = no split.")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="Which contiguous shard [0..num-shards-1] THIS "
                         "process handles. Each shard writes its own run "
                         "dir; merge with merge_shards.py.")
    ap.add_argument("--icl-dir", default=str(DEFAULT_ICL_DIR), type=Path)
    ap.add_argument("--photos-dir", default=str(DEFAULT_TEST_PHOTOS), type=Path)
    ap.add_argument("--descriptions-dir", default=str(DEFAULT_TEST_DESCRIPTIONS), type=Path)
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=Path)
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Limit eval to first N test samples (for smoke testing).")
    ap.add_argument("--max-new-tokens", type=int, default=2048,
                    help="Upper bound on generated tokens. ICL gold outputs are "
                         "591-878 tokens (avg ~690); 2048 gives ~3x headroom. "
                         "Per-sample time only grows if model rambles past EOS.")
    ap.add_argument("--save-every", type=int, default=20,
                    help="Checkpoint partial results to disk every N samples.")
    ap.add_argument("--run-tag", default=None,
                    help="Subfolder name under output-dir; default uses model name + timestamp.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # --- Output dir ---
    if args.run_tag:
        run_tag = args.run_tag
    else:
        precision = "int4" if args.load_in_4bit else args.dtype
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_tag = f"{args.model_name.split('/')[-1]}__icl6__{precision}__{timestamp}"
    if args.num_shards > 1:
        run_tag = f"{run_tag}__shard{args.shard_id}of{args.num_shards}"
    run_dir = args.output_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Output dir: %s", run_dir)

    # --- Load data ---
    icl_examples = load_icl_examples(args.icl_dir)
    logging.info("Loaded %d in-context examples from %s", len(icl_examples), args.icl_dir)
    for ex in icl_examples:
        logging.info("  ICL %s -> %s", ex["id"], ex["diagnosis"])

    test_records = load_test_records(args.photos_dir, args.descriptions_dir)
    logging.info("Loaded %d test records", len(test_records))
    if args.num_samples:
        test_records = test_records[:args.num_samples]
        logging.info("Restricted to first %d", len(test_records))
    if args.num_shards > 1:
        n = len(test_records)
        per = (n + args.num_shards - 1) // args.num_shards
        lo = args.shard_id * per
        hi = min(n, lo + per)
        test_records = test_records[lo:hi]
        logging.info("SHARD %d/%d -> records [%d:%d] = %d samples",
                     args.shard_id, args.num_shards, lo, hi,
                     len(test_records))

    # --- Save run config ---
    config_summary = {
        "model_name": args.model_name,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "attn_implementation": args.attn_implementation,
        "icl_dir": str(args.icl_dir),
        "icl_count": len(icl_examples),
        "icl_ids": [ex["id"] for ex in icl_examples],
        "photos_dir": str(args.photos_dir),
        "descriptions_dir": str(args.descriptions_dir),
        "num_test_samples": len(test_records),
        "max_new_tokens": args.max_new_tokens,
        "system_prompt": SYSTEM_PROMPT,
        "test_query_text": TEST_QUERY_TEXT,
        "icl_user_text_source": "JSON 'input' field per example (matches Grace's notebook)",
        "backend": args.backend,
        "tensor_parallel_size": (args.tensor_parallel_size
                                 if args.backend == "vllm" else None),
        "greedy": args.greedy,
        "device_map": (args.device_map if args.backend == "hf" else None),
    }
    (run_dir / "config.json").write_text(json.dumps(config_summary, indent=2, ensure_ascii=False))

    # --- Load model ---
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_load = time.time()
    if args.backend == "vllm":
        logging.info("backend=vllm  tensor_parallel_size=%d",
                     args.tensor_parallel_size)
        model, processor = load_vllm(
            args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype)
    else:
        logging.info("backend=hf  device_map=%s greedy=%s",
                     args.device_map, args.greedy)
        model, processor = load_model(
            args.model_name,
            dtype=args.dtype,
            load_in_4bit=args.load_in_4bit,
            device_map=args.device_map,
            attn_implementation=args.attn_implementation,
        )
    load_seconds = time.time() - t_load
    logging.info("Model loaded in %.1fs", load_seconds)
    if torch.cuda.is_available():
        vram_gb = sum(torch.cuda.memory_allocated(i)
                      for i in range(torch.cuda.device_count())) / 1024**3
        logging.info("VRAM after load: %.2f GB", vram_gb)

    # --- Inference loop (with auto-resume from existing predictions.jsonl) ---
    pred_path = run_dir / "predictions.jsonl"

    # AUTO-RESUME: if predictions.jsonl already exists in this run_dir, load it
    # and skip those IDs. Append new samples to the same file.
    #
    # SAFETY POLICY: we ALWAYS drop the last line of any existing predictions.jsonl
    # (regardless of whether it parses cleanly) and re-run that sample. Reasoning:
    # the last line is the one most likely to have been written mid-flush when the
    # user Ctrl-C'd, so it could be truncated or otherwise corrupt. Cheaper to
    # just re-run one sample than to silently keep a bad row.
    results: list[dict] = []
    done_ids: set[str] = set()
    n_loaded = 0
    n_dropped_last = 0
    n_malformed_skipped = 0
    if pred_path.exists():
        lines = pred_path.read_text(encoding="utf-8").splitlines()
        # Drop the last non-empty entry unconditionally (re-run it).
        # We rewrite the file to remove that line so we don't end up with a
        # duplicate row when the re-run appends.
        complete_lines: list[str] = []
        if lines:
            last_idx = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
            if last_idx >= 0:
                complete_lines = lines[:last_idx]
                n_dropped_last = 1
            else:
                complete_lines = lines  # all blank, nothing to drop
        # Truncate file: keep only the kept lines (with trailing newline if non-empty)
        pred_path.write_text(
            ("\n".join(complete_lines) + "\n") if complete_lines else "",
            encoding="utf-8",
        )
        for line in complete_lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("id"):
                    results.append(r)
                    done_ids.add(str(r["id"]))
                    n_loaded += 1
            except json.JSONDecodeError:
                n_malformed_skipped += 1
                logging.warning("Skipping malformed line in existing predictions.jsonl")
        if n_loaded or n_dropped_last or n_malformed_skipped:
            logging.info(
                "🔁 RESUMING from %s",
                pred_path,
            )
            logging.info("   Kept as done:        %d samples", n_loaded)
            logging.info("   Dropped last entry:  %d (will re-run for safety)", n_dropped_last)
            if n_malformed_skipped:
                logging.warning("   Skipped malformed:  %d", n_malformed_skipped)
            if done_ids:
                logging.info("   First 5 done IDs:    %s", sorted(done_ids)[:5])

    remaining = [r for r in test_records if str(r["id"]) not in done_ids]
    if not remaining:
        logging.info("All %d test records are already done — nothing to do.", len(test_records))
    else:
        logging.info("Will process %d remaining samples (%d already done out of %d total)",
                     len(remaining), len(done_ids), len(test_records))

    # vLLM: submit the whole (sharded) batch ONCE up front so continuous
    # batching runs them concurrently. The per-record loop below then just
    # looks up the answer — keeps ALL the resume/write/parse/metrics logic
    # (and the hf path) completely unchanged.
    _vllm_cache: dict = {}
    if args.backend == "vllm" and remaining:
        logging.info("vLLM: batch-submitting %d prompts (continuous "
                     "batching across TP=%d)…",
                     len(remaining), args.tensor_parallel_size)
        _bm = [build_messages(icl_examples, r["image_path"])
               for r in remaining]
        _texts = vllm_generate_batch(model, processor, _bm,
                                     max_new_tokens=args.max_new_tokens,
                                     greedy=args.greedy)
        _vllm_cache = {str(r["id"]): t for r, t in zip(remaining, _texts)}
        logging.info("vLLM: batch complete (%d outputs)", len(_vllm_cache))

    t_infer = time.time()
    pbar = tqdm(total=len(test_records), initial=len(done_ids), desc="ICL eval")
    for rec in remaining:
        msgs = build_messages(icl_examples, rec["image_path"])
        try:
            if args.backend == "vllm":
                generated = _vllm_cache.get(str(rec["id"]), "")
            else:
                generated = generate_one(model, processor, msgs,
                                         max_new_tokens=args.max_new_tokens,
                                         greedy=args.greedy)
        except Exception as e:
            logging.exception("Sample %s failed: %s", rec["id"], e)
            generated = ""
        pred, parse_status = parse_diagnosis(generated)
        result = {
            "id": rec["id"],
            "true_diagnosis": rec["true_diagnosis"],
            "pred_diagnosis": pred,           # 'not likely' / 'borderline' / 'likely' / 'NO_DIAGNOSIS_FOUND'
            "parse_status": parse_status,     # 'ok_anchor' / 'ok_fallback' / 'no_diagnosis_found'
            "generated": generated,           # raw model output, for re-analysis later
        }
        results.append(result)
        with pred_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if len(results) % args.save_every == 0:
            _save_pickle(results, run_dir / "predictions.pkl")
        pbar.update(1)
    pbar.close()

    n_processed = len(remaining)
    infer_seconds = time.time() - t_infer
    if n_processed > 0:
        logging.info("Inference took %.1fs for %d new samples (%.2fs per sample)",
                     infer_seconds, n_processed, infer_seconds / n_processed)
    else:
        logging.info("Inference loop ran in %.1fs (no new samples processed).", infer_seconds)

    # --- Final save ---
    # Per user preference: this script ONLY produces raw outputs.
    # All metric/statistics analysis is done by a SEPARATE user-written script.
    _save_pickle(results, run_dir / "predictions.pkl")

    # Tiny run summary (not metrics — just runtime/sanity info)
    from collections import Counter
    summary = {
        "model_name":           args.model_name,
        "num_samples_total":    len(results),         # includes resumed
        "num_samples_new_run":  n_processed,          # only this invocation
        "num_samples_resumed":  len(done_ids),        # already in jsonl from prior run
        "load_seconds":         round(load_seconds, 2),
        "infer_seconds_new":    round(infer_seconds, 2),
        "seconds_per_sample_new": round(infer_seconds / max(1, n_processed), 4) if n_processed else None,
        "parse_status_counts":  dict(Counter(r.get("parse_status", "") for r in results)),
        "pred_distribution":    dict(Counter(r.get("pred_diagnosis", "") for r in results)),
    }
    if torch.cuda.is_available():
        peak_vram = sum(torch.cuda.max_memory_allocated(i)
                        for i in range(torch.cuda.device_count())) / 1024**3
        summary["peak_vram_gb"] = round(peak_vram, 3)
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Run summary: %s", json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Raw outputs in: %s/predictions.{jsonl,pkl}", run_dir)


def _save_pickle(results, path: Path):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_pickle(path)


if __name__ == "__main__":
    main()
