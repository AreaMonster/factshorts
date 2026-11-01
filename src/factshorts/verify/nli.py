"""Local NLI (entailment) check with a DeBERTa-v3 cross-encoder on CPU."""

from __future__ import annotations

import logging
import re
from typing import Protocol

from factshorts.models import CheckResult
from factshorts.verify.normalize import normalize_chars, split_sentences

log = logging.getLogger(__name__)


class NLIModel(Protocol):
    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Probabilities for 'entailment', 'neutral', 'contradiction'."""
        ...


class TransformersNLI:
    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self.labels = {i: lab.lower() for i, lab in self.model.config.id2label.items()}

    def scores(self, premise: str, hypothesis: str) -> dict[str, float]:
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation="only_first",
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**inputs.to(self.device)).logits[0]
        probs = self.torch.softmax(logits, dim=-1).tolist()
        return {self.labels[i]: p for i, p in enumerate(probs)}


def _resolve_pronoun(text: str, title: str) -> str:
    """'It is …' / 'Its …' → '<title> is …' / "<title>'s …" so the pair is self-contained."""
    text = normalize_chars(text)
    text = re.sub(r"^Its\b", f"{title}'s", text)
    text = re.sub(r"^It's\b", f"{title} is", text)
    return re.sub(r"^It\b", title, text)


def premises(quote: str, paragraph: str | None, title: str, window: int = 2) -> list[str]:
    """The quote alone, the quote with its neighbors, and the whole paragraph if short."""
    quote = _resolve_pronoun(quote, title)
    out = [quote]
    if paragraph:
        sentences = split_sentences(paragraph)
        q = normalize_chars(quote).casefold()
        idx = next(
            (
                i
                for i, s in enumerate(sentences)
                if normalize_chars(s).casefold() in q or q in normalize_chars(s).casefold()
            ),
            None,
        )
        if idx is not None:
            ctx = sentences[max(0, idx - window) : idx + window + 1]
            out.append(f"{title}. " + " ".join(ctx))
        if len(paragraph.split()) <= 300:
            out.append(f"{title}. {normalize_chars(paragraph)}")
    return list(dict.fromkeys(out))


def nli_check(
    model: NLIModel, claim: str, quote: str, paragraph: str | None, title: str, threshold: float
) -> CheckResult:
    hypothesis = _resolve_pronoun(claim, title)
    best: dict[str, float] = {}
    for premise in premises(quote, paragraph, title):
        s = model.scores(premise, hypothesis)
        if s.get("entailment", 0.0) > best.get("entailment", -1.0):
            best = s
        if best["entailment"] >= threshold:
            break  # narrowest premise that entails the claim is enough
    ent = best.get("entailment", 0.0)
    con = best.get("contradiction", 0.0)
    if ent >= threshold:
        return CheckResult(
            check="nli", status="pass", score=round(ent, 3), reason=f"entailment {ent:.2f}"
        )
    kind = "contradicted" if con > ent else "not entailed"
    return CheckResult(
        check="nli",
        status="fail",
        score=round(ent, 3),
        reason=f"{kind} by the source (entailment {ent:.2f}, contradiction "
        f"{con:.2f}; threshold {threshold:.2f})",
    )
