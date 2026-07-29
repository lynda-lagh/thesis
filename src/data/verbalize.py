"""Turn a raw (head, relation, tail) triple into natural-language text and into a
True/False/Unknown instruction prompt for Qwen2.5-Instruct.

Kept dependency-free so it can run anywhere (sandbox, Kaggle, laptop).
"""
from __future__ import annotations

import re

# ---- surface realization -----------------------------------------------------

def humanize_entity(e: str) -> str:
    """`Albert_Einstein` -> `Albert Einstein`; strips YAGO/Wikidata decorations."""
    e = e.strip().strip("<>")
    e = e.split("/")[-1]              # drop any namespace prefix
    e = e.replace("_", " ").strip()
    return e


def humanize_relation(r: str) -> str:
    """`wasBornIn` -> `was born in`; `playsFor` -> `plays for`."""
    r = r.strip().strip("<>").split("/")[-1]
    r = r.lstrip("_")
    # split camelCase and snake_case
    r = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", r)
    r = r.replace("_", " ")
    return r.lower().strip()


def verbalize_triple(head: str, relation: str, tail: str) -> str:
    """Readable sentence for a triple, e.g. 'Albert Einstein was born in Ulm.'"""
    return f"{humanize_entity(head)} {humanize_relation(relation)} {humanize_entity(tail)}."


# ---- instruction prompt ------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful knowledge-graph reasoning assistant. You judge whether a "
    "statement is true using world knowledge. Knowledge graphs are open-world: a "
    "fact that is merely missing is Unknown, not False. Only answer False when the "
    "statement contradicts a known fact. When there is not enough evidence to "
    "decide, answer Unknown."
)

USER_TEMPLATE = (
    "Statement: {sentence}\n\n"
    "Is this statement True, False, or Unknown? "
    "Answer with exactly one word: True, False, or Unknown."
)


def build_prompt(head: str, relation: str, tail: str) -> dict:
    """Return a chat-format prompt (list of messages) + the flat user string.

    The training/inference scripts apply the model's chat template to `messages`.
    """
    sentence = verbalize_triple(head, relation, tail)
    user = USER_TEMPLATE.format(sentence=sentence)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return {"messages": messages, "user": user, "sentence": sentence}
