"""
LLM auto-verifier: given a claimed (GO term, concept) for an SAE feature and
the real-world identity of its top-activating proteins, produce a
confirmed/plausible/spurious/unverifiable judgment via an LLM call, following
a fixed decision procedure (see `_RUBRIC` below).

Verification is a necessary statistical stage in interpretability
evaluation: raw feature-activation claims need a scalable way to be checked
against ground truth. This module is that scalable, automated protocol. It
is infrastructure, not the claim itself: the verifier's own accuracy must be
calibrated against two-human-annotator consensus before its output can be
trusted as a sweep curve, and its run-to-run consistency (see
`consistency_rate` below) must be reported alongside its accuracy: a
"validated, scalable" claim is incomplete without both.

Fetches real protein records from the public UniProt REST API (no auth
required) — never asks the LLM to recall a protein's function from memory,
which would reintroduce exactly the pretraining-familiarity ("fame") bias
that undermines the verifier's validity as an independent check.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

_UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"

VALID_STATUSES = {"confirmed", "plausible", "spurious", "unverifiable"}
VALID_CONFIDENCE = {"high", "medium", "low"}


@dataclass
class ProteinRecord:
    """A real protein's documented identity, fetched from UniProt — never
    from LLM memory (see module docstring on the "fame" bias risk)."""

    accession: str
    recommended_name: str
    go_ids: list[str] = field(default_factory=list)
    go_names: list[str] = field(default_factory=list)  # from GO cross-ref "properties", if present

    @property
    def has_go_annotations(self) -> bool:
        return len(self.go_ids) > 0


@dataclass
class VerifierJudgment:
    status: str
    confidence: str
    reasoning: str
    raw_response: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status {self.status!r}; must be one of {VALID_STATUSES}")
        if self.confidence not in VALID_CONFIDENCE:
            raise ValueError(
                f"Invalid confidence {self.confidence!r}; must be one of {VALID_CONFIDENCE}"
            )


class VerifierResponseError(ValueError):
    """Raised when the LLM's response can't be parsed into a valid
    VerifierJudgment — surfaced explicitly rather than silently guessing a
    status, since a mis-parsed response masquerading as a real judgment
    would corrupt the calibration data this whole pipeline exists to
    produce trustworthy numbers from."""


def fetch_protein_record(accession: str, timeout: float = 15.0) -> ProteinRecord:
    """
    Fetch a protein's real documented identity from the public UniProt REST
    API. Returns a record with empty go_ids (not an exception) if the entry
    has no GO annotations — that is itself meaningful signal (the
    `unverifiable` category exists precisely for this case), not an error.
    """
    url = _UNIPROT_ENTRY_URL.format(accession=accession)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    name = (
        data.get("proteinDescription", {})
        .get("recommendedName", {})
        .get("fullName", {})
        .get("value")
    )
    if not name:
        # Uncharacterized/submitted-name-only entries lack a recommendedName.
        name = (
            data.get("proteinDescription", {})
            .get("submissionNames", [{}])[0]
            .get("fullName", {})
            .get("value", "(no documented name)")
        )

    go_ids: list[str] = []
    go_names: list[str] = []
    for xref in data.get("uniProtKBCrossReferences", []):
        if xref.get("database") != "GO":
            continue
        go_ids.append(xref["id"])
        for prop in xref.get("properties", []):
            if prop.get("key") == "GoTerm":
                go_names.append(prop.get("value", ""))

    return ProteinRecord(
        accession=accession, recommended_name=name, go_ids=go_ids, go_names=go_names,
    )


_RUBRIC = """\
You are verifying a claim about a sparse-autoencoder (SAE) feature trained on a protein \
language model. The feature's single-feature AUROC against a GO-term label set produced a \
CANDIDATE claim that the feature encodes a specific biological concept. High AUROC alone is \
NOT proof — it may reflect selection bias (winner's curse: the best of many noisy per-feature \
estimates is upward-biased) or a feature that tracks a broader structural class rather than \
the specific claimed concept. Your job is to check the claim against the REAL, documented \
identity of the feature's top-activating proteins (provided below, fetched directly from \
UniProt — not from your own training-data memory of these proteins).

Decide among exactly four status categories, in this precedence order (stop at the first that applies):

CONFIRMED: the claimed GO term (or an explicit, documented synonym/sub-type) appears directly \
in the real protein's own GO annotation list, or its documented molecular function is an \
unambiguous, specific match to the claim — not just a broad structural resemblance.

PLAUSIBLE: one of three sub-cases (state which one applies):
  (a) Right class, wrong specificity — the real protein is a confirmed, structurally correct \
member of the broad category implied by the claim, but the specific instance doesn't match \
(e.g. a genuine Class A GPCR when the claim was "serotonin receptor," but the real receptor \
binds a different ligand).
  (b) Functionally consistent, not explicitly documented — the real protein's known biology is \
mechanistically consistent with the claim, but the specific GO term wasn't in the fetched \
record (may reflect UniProt annotation incompleteness, not a true mismatch).
  (c) Moonlighting/multi-function — the real protein has multiple independently-documented \
functions, and the claim matches one of them but not the protein's primary characterized role.

SPURIOUS: the real protein's documented function bears no defensible relationship to the \
claim, even loosely — not the same broad structural class, not mechanistically related, not a \
plausible annotation gap.

UNVERIFIABLE: the real top-activating protein(s) have NO GO annotations at all to check the \
claim against. Do not force this into SPURIOUS for lack of a clean negative, or into \
PLAUSIBLE for lack of a clean positive.

Respond with ONLY a JSON object (no markdown fences, no other text), with exactly these keys:
{"status": "confirmed"|"plausible"|"spurious"|"unverifiable", \
"confidence": "high"|"medium"|"low", "reasoning": "one to three sentences"}
"""


def build_verification_prompt(
    claimed_concept: str,
    claimed_go: list[str],
    protein_records: list[ProteinRecord],
) -> str:
    """Build the exact prompt sent to the LLM verifier, following the
    decision procedure in `_RUBRIC` verbatim."""
    records_block = "\n".join(
        f"- {r.accession}: \"{r.recommended_name}\" — "
        f"GO annotations: {', '.join(f'{gid} ({gname})' for gid, gname in zip(r.go_ids, r.go_names)) if r.go_ids else 'NONE'}"
        for r in protein_records
    )
    return (
        f"{_RUBRIC}\n"
        f"CLAIM:\n"
        f"  Claimed GO term(s): {', '.join(claimed_go)}\n"
        f"  Claimed concept: {claimed_concept}\n\n"
        f"REAL TOP-ACTIVATING PROTEINS (from UniProt, not your memory):\n"
        f"{records_block}\n"
    )


def parse_verifier_response(text: str) -> VerifierJudgment:
    """
    Parse the LLM's response into a VerifierJudgment. Tolerates the response
    being wrapped in a markdown code fence (```json ... ```), a common LLM
    formatting habit even when explicitly told not to. Raises
    VerifierResponseError (not a silent fallback status) on anything else
    unparseable — a mis-parsed response must never masquerade as a real
    judgment in the calibration data.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise VerifierResponseError(f"Could not parse LLM response as JSON: {text!r}") from exc

    missing = {"status", "confidence", "reasoning"} - parsed.keys()
    if missing:
        raise VerifierResponseError(f"LLM response missing required keys {missing}: {text!r}")

    try:
        return VerifierJudgment(
            status=parsed["status"],
            confidence=parsed["confidence"],
            reasoning=parsed["reasoning"],
            raw_response=text,
        )
    except ValueError as exc:
        raise VerifierResponseError(str(exc)) from exc


def call_llm_verifier(
    prompt: str,
    model: str = "claude-sonnet-5",
    api_key: str | None = None,
    max_tokens: int = 512,
) -> VerifierJudgment:
    """
    Call the Anthropic API to get one verification judgment. Requires
    ANTHROPIC_API_KEY (or an explicit api_key) — raises immediately with an
    actionable message if neither is set, rather than failing deep inside
    an opaque SDK stack trace.
    """
    import os

    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Set the ANTHROPIC_API_KEY environment "
            "variable or pass api_key= explicitly to call_llm_verifier()."
        )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return parse_verifier_response(text)


def verify_feature(
    claimed_concept: str,
    claimed_go: list[str],
    accessions: list[str],
    n_repeats: int = 1,
    model: str = "claude-sonnet-5",
    api_key: str | None = None,
) -> list[VerifierJudgment]:
    """
    End-to-end verification of one feature's claim: fetch real UniProt
    records for its top-activating proteins, build the prompt, and call the
    LLM verifier `n_repeats` times (for run-to-run consistency checking —
    see `consistency_rate` — a "validated, scalable" claim needs the
    verifier to agree with itself, not just with humans).
    """
    records = [fetch_protein_record(acc) for acc in accessions]
    prompt = build_verification_prompt(claimed_concept, claimed_go, records)
    return [
        call_llm_verifier(prompt, model=model, api_key=api_key) for _ in range(n_repeats)
    ]


@dataclass
class TextExampleRecord:
    """A real Bias-in-Bios biography's documented profession label and text
    — the non-biology analog of ProteinRecord. No fetch step
    is needed here (unlike fetch_protein_record's UniProt call): the "real,
    documented identity" of a Bias-in-Bios example is already fully in hand
    once biolens.data.saebench_datasets.load_bias_in_bios has loaded the
    dataset — the profession label and the biography text are both already
    local data, not something to look up externally."""

    example_id: str
    text: str
    documented_professions: list[str] = field(default_factory=list)

    @property
    def has_documented_profession(self) -> bool:
        return len(self.documented_professions) > 0


_NONBIO_RUBRIC = """\
You are verifying a claim about a sparse-autoencoder (SAE) feature trained on a pretrained, \
non-biology language model (Gemma Scope on Gemma-2-2B, used here as a non-biology control \
domain). The feature's single-feature AUROC against a profession- \
classification label set (Bias-in-Bios) produced a CANDIDATE claim that the feature encodes a \
specific profession concept. High AUROC alone is NOT proof — it may reflect selection bias \
(winner's curse: the best of many noisy per-feature estimates is upward-biased) or a feature that \
tracks a demographic/stylistic correlate of the profession rather than the profession itself. \
Your job is to check the claim against the REAL biography text and its dataset-provided \
profession label (provided below — the actual ground truth for this example, not your own \
impression of what profession the writing style "sounds like").

Decide among exactly four status categories, in this precedence order (stop at the first that applies):

CONFIRMED: the claimed profession matches the documented profession label exactly (or an \
explicit synonym, e.g. "attorney" vs "lawyer"), AND the biography text itself contains specific, \
profession-relevant content supporting it — not just a bare label match with generic text.

PLAUSIBLE: one of three sub-cases (state which one applies):
  (a) Right class, wrong specificity — the documented profession is a genuine, closely related \
sub-field or parent category of the claim (e.g. claimed "surgeon," documented "physician").
  (b) Text-consistent, label ambiguous — the biography's actual content is consistent with the \
claim even though the exact documented label doesn't cleanly match (may reflect dataset labeling \
granularity, not a true mismatch).
  (c) Confound present — the biography contains genuine profession-relevant content matching the \
claim, but ALSO contains a strong, independent confound (e.g. gender-coded phrasing, distinctive \
writing-style artifacts) that could equally explain why this specific example activates the \
feature — the profession match is real but may not be the true driver of activation.

SPURIOUS: the documented profession bears no defensible relationship to the claim, and the \
biography's actual text content gives no profession-specific support for it either — the \
feature's activation on this example looks driven by something else entirely (e.g. demographic \
or stylistic correlates unrelated to profession content).

UNVERIFIABLE: the biography text is too short, generic, or truncated to determine \
profession-specific content one way or the other, or its documented profession label is missing.

Respond with ONLY a JSON object (no markdown fences, no other text), with exactly these keys:
{"status": "confirmed"|"plausible"|"spurious"|"unverifiable", \
"confidence": "high"|"medium"|"low", "reasoning": "one to three sentences"}
"""


def build_text_verification_prompt(
    claimed_concept: str,
    example_records: list[TextExampleRecord],
) -> str:
    """Build the exact prompt sent to the LLM verifier for a non-biology
    feature claim — the text-domain analog of
    build_verification_prompt."""
    records_block = "\n".join(
        f"- Example {r.example_id}: documented profession(s): "
        f"{', '.join(r.documented_professions) if r.documented_professions else 'NONE'}\n"
        f'  Biography text: "{r.text}"'
        for r in example_records
    )
    return (
        f"{_NONBIO_RUBRIC}\n"
        f"CLAIM:\n"
        f"  Claimed profession: {claimed_concept}\n\n"
        f"REAL TOP-ACTIVATING BIOGRAPHY EXAMPLES (from the Bias-in-Bios dataset):\n"
        f"{records_block}\n"
    )


def verify_text_feature(
    claimed_concept: str,
    example_records: list[TextExampleRecord],
    n_repeats: int = 1,
    model: str = "claude-sonnet-5",
    api_key: str | None = None,
) -> list[VerifierJudgment]:
    """
    End-to-end verification of one non-biology feature's claim — the
    text-domain analog of verify_feature. Unlike verify_feature, this takes
    already-built example_records directly rather than fetching them (there
    is no external "real record" source to fetch from for Bias-in-Bios —
    see TextExampleRecord's docstring); the caller is responsible for
    identifying a feature's top-activating example IDs and their documented
    profession label/text (e.g. from a verified_features registry's
    evidence_example_ids plus biolens.data.saebench_datasets.load_bias_in_bios).
    """
    prompt = build_text_verification_prompt(claimed_concept, example_records)
    return [
        call_llm_verifier(prompt, model=model, api_key=api_key) for _ in range(n_repeats)
    ]


def consistency_rate(judgments: list[VerifierJudgment]) -> float:
    """
    Fraction of judgments matching the modal (most common) status — the
    run-to-run consistency metric for repeated calls on the same feature.
    Returns 1.0 for a single judgment (trivially self-consistent) and 0.0
    for an empty list (nothing to be consistent about).
    """
    if not judgments:
        return 0.0
    counts = Counter(j.status for j in judgments)
    modal_count = counts.most_common(1)[0][1]
    return modal_count / len(judgments)
