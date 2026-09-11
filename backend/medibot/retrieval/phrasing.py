"""Wording shared by both refusal paths, so the two read as one voice.

A user who is turned away from billing documents and one who is turned away from the
claims records should get the same shape of sentence: what is closed, then what is open.
Keeping the phrasing in one place is what stops the two drifting apart.
"""

from __future__ import annotations

from medibot.config import ROLE_COLLECTIONS

# How the operations database is named to a reader. The frontend rail uses the same words.
RECORDS = "the claims and ticket records"


def english_list(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def describe_role(role: str) -> str:
    """'billing_executive' -> 'a billing executive'."""
    words = role.replace("_", " ")
    return f"{'an' if words[0] in 'aeiou' else 'a'} {words}"


def refusal(role: str, blocked: str, inferred: bool = False) -> str:
    """What is closed, then what is still open. Never only the first half: a reader told
    what they cannot have and not what they can has been given half an answer.

    `inferred` hedges the first clause. Which collection a question belongs to is a
    classifier's guess, so stating it flatly claims more than we checked; the role gate on
    the records is a fact and needs no hedge.
    """
    opening = (
        f"This looks like a question for {blocked}, which {describe_role(role)} cannot read."
        if inferred
        else f"As {describe_role(role)}, you don't have access to {blocked}."
    )
    return (
        f"{opening} "
        f"I can only answer questions from the {english_list(ROLE_COLLECTIONS[role])} collections."
    )
