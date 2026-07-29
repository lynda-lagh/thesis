"""Prompt-engineering phase: variant prompt builders for the FROZEN base model.

This is the no-training baseline: same benchmark, same base model, no QLoRA -
just different ways of asking. It gives the comparison point for "was
fine-tuning worth it", and the winning variant's discriminative distribution
is exactly the kind of P(True)-style signal Step 3 reads from the fine-tuned
model, so this phase reuses the same verbalization and scoring machinery.

Variants
--------
  zero_shot : the same instruction/question format used for fine-tuning
              (Step 1's verbalize.py), asked of the frozen model directly.
  few_shot  : zero-shot + a handful of worked (statement -> label) examples,
              one per class, drawn from the TRAIN split (never test) so there
              is no leakage.
  evidence  : zero-shot + a short list of the head entity's OTHER known
              triples from the training graph - the "activate Stage 2"
              instantiation: give the model retrieved structural context
              instead of just the bare triple.
  cot       : ask the model to reason step by step before answering, ending
              with a forced 'Final answer: <label>' line so the answer can
              still be parsed and re-scored discriminatively.
"""
from __future__ import annotations

import random

from src.data.verbalize import SYSTEM_PROMPT, humanize_entity, humanize_relation, verbalize_triple

FEWSHOT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " Here are worked examples showing the expected answer format."
)

EVIDENCE_SYSTEM_PROMPT = (
    "You are a careful knowledge-graph reasoning assistant. You are given some "
    "known facts about an entity, then a statement to judge. Knowledge graphs "
    "are open-world: a fact that is merely missing is Unknown, not False. Use "
    "the known facts as evidence; only answer False when the statement "
    "contradicts them. When the known facts do not settle the question, "
    "answer Unknown."
)

COT_SYSTEM_PROMPT = (
    "You are a careful knowledge-graph reasoning assistant. Think step by "
    "step about whether the statement is True, False, or Unknown, using the "
    "open-world assumption: a fact that is merely missing is Unknown, not "
    "False. End your answer with a final line in EXACTLY this format:\n"
    "Final answer: <True|False|Unknown>"
)

QUESTION_TEMPLATE = (
    "Statement: {sentence}\n\n"
    "Is this statement True, False, or Unknown? "
    "Answer with exactly one word: True, False, or Unknown."
)

COT_QUESTION_TEMPLATE = (
    "Statement: {sentence}\n\n"
    "Think step by step, then answer.\n"
    "Final answer: <True|False|Unknown>"
)


# --------------------------------------------------------------------------- #
# Variant builders - each returns {"messages": [...]} like verbalize.build_prompt
# --------------------------------------------------------------------------- #

def build_zero_shot(head: str, relation: str, tail: str) -> dict:
    sentence = verbalize_triple(head, relation, tail)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": QUESTION_TEMPLATE.format(sentence=sentence)},
        ],
        "sentence": sentence,
    }


def build_few_shot(head: str, relation: str, tail: str, exemplars: list[dict]) -> dict:
    """exemplars: list of {"sentence":..., "label":...} drawn from TRAIN only."""
    sentence = verbalize_triple(head, relation, tail)
    messages = [{"role": "system", "content": FEWSHOT_SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({"role": "user", "content": QUESTION_TEMPLATE.format(sentence=ex["sentence"])})
        messages.append({"role": "assistant", "content": ex["label"]})
    messages.append({"role": "user", "content": QUESTION_TEMPLATE.format(sentence=sentence)})
    return {"messages": messages, "sentence": sentence}


def build_evidence(head: str, relation: str, tail: str, neighbor_sentences: list[str]) -> dict:
    sentence = verbalize_triple(head, relation, tail)
    if neighbor_sentences:
        facts = " ".join(neighbor_sentences)
        user = f"Known facts: {facts}\n\n" + QUESTION_TEMPLATE.format(sentence=sentence)
    else:
        user = "Known facts: (none found)\n\n" + QUESTION_TEMPLATE.format(sentence=sentence)
    return {
        "messages": [
            {"role": "system", "content": EVIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "sentence": sentence,
    }


def build_cot(head: str, relation: str, tail: str) -> dict:
    sentence = verbalize_triple(head, relation, tail)
    return {
        "messages": [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": COT_QUESTION_TEMPLATE.format(sentence=sentence)},
        ],
        "sentence": sentence,
    }


VARIANTS = {"zero_shot": build_zero_shot, "few_shot": build_few_shot,
            "evidence": build_evidence, "cot": build_cot}


# --------------------------------------------------------------------------- #
# Support pools: few-shot exemplars + neighbor evidence, built from TRAIN only
# --------------------------------------------------------------------------- #

def pick_fewshot_exemplars(train_rows: list[dict], k_per_class: int = 1, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_label = {"True": [], "False": [], "Unknown": []}
    for r in train_rows:
        by_label[r["label"]].append(r)
    exemplars = []
    for label, rows in by_label.items():
        rng.shuffle(rows)
        for r in rows[:k_per_class]:
            exemplars.append({"sentence": r["sentence"], "label": label})
    rng.shuffle(exemplars)
    return exemplars


def neighbor_evidence(head: str, relation: str, kg, max_facts: int = 3) -> list[str]:
    """Verbalize up to max_facts OTHER known triples with this head (any
    relation, excluding the exact (head,relation) pair being judged is not
    required - showing the head's other facts is exactly the 'retrieved
    neighborhood' Stage-2 evidence)."""
    sentences = []
    for r2, tails in [(r2, ts) for (h2, r2), ts in kg.ht.items() if h2 == head]:
        for t2 in tails:
            sentences.append(verbalize_triple(head, r2, t2))
            if len(sentences) >= max_facts:
                return sentences
    return sentences
