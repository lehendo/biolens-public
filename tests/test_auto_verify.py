"""
Tests for the LLM auto-verifier (the scalable verification infrastructure
in biolens.eval.auto_verify).

Real (network) tests against the actual UniProt REST API use accessions from
the existing configs/verified_features registry — this doubles as a
regression check that the fetcher agrees with the known-correct manual
verification already on record for those exact proteins.

LLM calls themselves are mocked throughout (no ANTHROPIC_API_KEY available
in CI/this environment) — only the prompt-construction, response-parsing,
and consistency-checking logic is under test there, following the same
mocking pattern test_uniprot.py uses for network/subprocess boundaries.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── UniProt fetching (real network calls — see module docstring) ─────────────

class TestFetchProteinRecord:
    def test_known_confirmed_feature_protein(self):
        """P31383 (feature 1103's evidence) is documented in the registry as
        PP2A regulatory subunit A, with GO:0000159 ('protein phosphatase
        type 2A complex') directly annotated — this is the registry's own
        `confirmed` worked example; the fetcher must reproduce it."""
        from biolens.eval.auto_verify import fetch_protein_record

        record = fetch_protein_record("P31383")
        assert "phosphatase" in record.recommended_name.lower()
        assert "GO:0000159" in record.go_ids

    def test_known_spurious_feature_protein(self):
        """Q47DI0 (feature 2253's evidence) is documented as KDO-8-P
        synthase, unrelated to the claimed aspartate transaminase activity —
        the registry's own `spurious` worked example."""
        from biolens.eval.auto_verify import fetch_protein_record

        record = fetch_protein_record("Q47DI0")
        assert "GO:0004069" not in record.go_ids  # the (wrong) claimed term

    def test_go_names_align_with_go_ids(self):
        from biolens.eval.auto_verify import fetch_protein_record

        record = fetch_protein_record("P31383")
        assert len(record.go_names) == len(record.go_ids)

    def test_has_go_annotations_property(self):
        from biolens.eval.auto_verify import fetch_protein_record

        record = fetch_protein_record("P31383")
        assert record.has_go_annotations is True


# ── Prompt construction ───────────────────────────────────────────────────────

class TestBuildVerificationPrompt:
    def _record(self, accession="P00001", name="Test protein", go_ids=None, go_names=None):
        from biolens.eval.auto_verify import ProteinRecord

        return ProteinRecord(
            accession=accession, recommended_name=name,
            go_ids=go_ids or [], go_names=go_names or [],
        )

    def test_includes_claimed_concept_and_go(self):
        from biolens.eval.auto_verify import build_verification_prompt

        prompt = build_verification_prompt(
            "acetylcholine receptor signaling", ["GO:0095500"], [self._record()],
        )
        assert "acetylcholine receptor signaling" in prompt
        assert "GO:0095500" in prompt

    def test_includes_all_four_status_categories(self):
        from biolens.eval.auto_verify import build_verification_prompt

        prompt = build_verification_prompt("x", ["GO:0000001"], [self._record()])
        for status in ["CONFIRMED", "PLAUSIBLE", "SPURIOUS", "UNVERIFIABLE"]:
            assert status in prompt

    def test_includes_protein_record_details(self):
        from biolens.eval.auto_verify import build_verification_prompt

        record = self._record(
            accession="P31383", name="PP2A regulatory subunit A",
            go_ids=["GO:0000159"], go_names=["protein phosphatase type 2A complex"],
        )
        prompt = build_verification_prompt("x", ["GO:0000001"], [record])
        assert "P31383" in prompt
        assert "PP2A regulatory subunit A" in prompt
        assert "GO:0000159" in prompt

    def test_protein_with_no_go_annotations_marked_none(self):
        from biolens.eval.auto_verify import build_verification_prompt

        record = self._record(go_ids=[], go_names=[])
        prompt = build_verification_prompt("x", ["GO:0000001"], [record])
        assert "NONE" in prompt

    def test_requests_json_only_response(self):
        from biolens.eval.auto_verify import build_verification_prompt

        prompt = build_verification_prompt("x", ["GO:0000001"], [self._record()])
        assert "JSON" in prompt
        assert "no markdown fences" in prompt.lower()


# ── Response parsing ──────────────────────────────────────────────────────────

class TestParseVerifierResponse:
    def test_parses_clean_json(self):
        from biolens.eval.auto_verify import parse_verifier_response

        text = '{"status": "confirmed", "confidence": "high", "reasoning": "Direct GO match."}'
        j = parse_verifier_response(text)
        assert j.status == "confirmed"
        assert j.confidence == "high"
        assert j.reasoning == "Direct GO match."

    def test_parses_json_wrapped_in_markdown_fence(self):
        from biolens.eval.auto_verify import parse_verifier_response

        text = '```json\n{"status": "spurious", "confidence": "high", "reasoning": "No relation."}\n```'
        j = parse_verifier_response(text)
        assert j.status == "spurious"

    def test_invalid_json_raises_verifier_response_error(self):
        from biolens.eval.auto_verify import VerifierResponseError, parse_verifier_response

        with pytest.raises(VerifierResponseError):
            parse_verifier_response("this is not json at all")

    def test_missing_required_key_raises(self):
        from biolens.eval.auto_verify import VerifierResponseError, parse_verifier_response

        with pytest.raises(VerifierResponseError):
            parse_verifier_response('{"status": "confirmed", "confidence": "high"}')

    def test_invalid_status_value_raises(self):
        from biolens.eval.auto_verify import VerifierResponseError, parse_verifier_response

        with pytest.raises(VerifierResponseError):
            parse_verifier_response(
                '{"status": "definitely_yes", "confidence": "high", "reasoning": "x"}'
            )

    def test_invalid_confidence_value_raises(self):
        from biolens.eval.auto_verify import VerifierResponseError, parse_verifier_response

        with pytest.raises(VerifierResponseError):
            parse_verifier_response(
                '{"status": "confirmed", "confidence": "very sure", "reasoning": "x"}'
            )


# ── VerifierJudgment validation ───────────────────────────────────────────────

class TestVerifierJudgment:
    def test_valid_construction(self):
        from biolens.eval.auto_verify import VerifierJudgment

        j = VerifierJudgment(status="plausible", confidence="medium", reasoning="x")
        assert j.status == "plausible"

    def test_rejects_invalid_status(self):
        from biolens.eval.auto_verify import VerifierJudgment

        with pytest.raises(ValueError):
            VerifierJudgment(status="maybe", confidence="high", reasoning="x")


# ── LLM call (mocked — no live API key in this environment) ──────────────────

class TestCallLlmVerifier:
    def test_raises_clear_error_without_api_key(self, monkeypatch):
        from biolens.eval.auto_verify import call_llm_verifier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            call_llm_verifier("some prompt")

    def test_calls_anthropic_client_and_parses_response(self):
        from biolens.eval.auto_verify import call_llm_verifier

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"status": "confirmed", "confidence": "high", "reasoning": "Match."}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client) as mock_anthropic:
            judgment = call_llm_verifier("test prompt", api_key="fake-key-for-test")

        mock_anthropic.assert_called_once_with(api_key="fake-key-for-test")
        mock_client.messages.create.assert_called_once()
        _, kwargs = mock_client.messages.create.call_args
        assert kwargs["messages"][0]["content"] == "test prompt"
        assert judgment.status == "confirmed"

    def test_uses_env_var_when_no_explicit_key(self, monkeypatch):
        from biolens.eval.auto_verify import call_llm_verifier

        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"status": "spurious", "confidence": "low", "reasoning": "x"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client) as mock_anthropic:
            call_llm_verifier("test prompt")

        mock_anthropic.assert_called_once_with(api_key="env-key")


# ── End-to-end verify_feature (mocked LLM, real UniProt fetch) ───────────────

class TestVerifyFeature:
    def test_end_to_end_with_real_uniprot_and_mocked_llm(self):
        from biolens.eval.auto_verify import verify_feature

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"status": "confirmed", "confidence": "high", "reasoning": "Direct match."}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("anthropic.Anthropic", return_value=mock_client):
            judgments = verify_feature(
                claimed_concept="protein phosphatase type 2A complex",
                claimed_go=["GO:0000159"],
                accessions=["P31383"],  # real UniProt fetch happens here
                n_repeats=3,
                api_key="fake-key-for-test",
            )

        assert len(judgments) == 3
        assert all(j.status == "confirmed" for j in judgments)
        assert mock_client.messages.create.call_count == 3


# ── Text-domain (non-biology control) prompt construction ────────────────────
# No network-fetch tests here (unlike TestFetchProteinRecord): TextExampleRecord
# is built directly from already-loaded Bias-in-Bios data, not fetched from an
# external API — see TextExampleRecord's docstring in auto_verify.py.

class TestBuildTextVerificationPrompt:
    def _record(self, example_id="biobio_0", text="She is a surgeon.", professions=None):
        from biolens.eval.auto_verify import TextExampleRecord

        return TextExampleRecord(
            example_id=example_id, text=text, documented_professions=professions or [],
        )

    def test_includes_claimed_concept(self):
        from biolens.eval.auto_verify import build_text_verification_prompt

        prompt = build_text_verification_prompt("surgeon", [self._record()])
        assert "surgeon" in prompt

    def test_includes_all_four_status_categories(self):
        from biolens.eval.auto_verify import build_text_verification_prompt

        prompt = build_text_verification_prompt("surgeon", [self._record()])
        for status in ["CONFIRMED", "PLAUSIBLE", "SPURIOUS", "UNVERIFIABLE"]:
            assert status in prompt

    def test_includes_example_details(self):
        from biolens.eval.auto_verify import build_text_verification_prompt

        record = self._record(
            example_id="biobio_42", text="She performed a triple bypass.",
            professions=["surgeon"],
        )
        prompt = build_text_verification_prompt("surgeon", [record])
        assert "biobio_42" in prompt
        assert "She performed a triple bypass." in prompt
        assert "surgeon" in prompt

    def test_example_with_no_documented_profession_marked_none(self):
        from biolens.eval.auto_verify import build_text_verification_prompt

        record = self._record(professions=[])
        prompt = build_text_verification_prompt("surgeon", [record])
        assert "NONE" in prompt

    def test_requests_json_only_response(self):
        from biolens.eval.auto_verify import build_text_verification_prompt

        prompt = build_text_verification_prompt("surgeon", [self._record()])
        assert "JSON" in prompt
        assert "no markdown fences" in prompt.lower()


class TestTextExampleRecord:
    def test_has_documented_profession_true(self):
        from biolens.eval.auto_verify import TextExampleRecord

        r = TextExampleRecord(example_id="x", text="t", documented_professions=["surgeon"])
        assert r.has_documented_profession is True

    def test_has_documented_profession_false(self):
        from biolens.eval.auto_verify import TextExampleRecord

        r = TextExampleRecord(example_id="x", text="t", documented_professions=[])
        assert r.has_documented_profession is False


# ── End-to-end verify_text_feature (mocked LLM, no network fetch needed) ─────

class TestVerifyTextFeature:
    def test_end_to_end_with_mocked_llm(self):
        from biolens.eval.auto_verify import TextExampleRecord, verify_text_feature

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"status": "confirmed", "confidence": "high", "reasoning": "Direct match."}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        records = [
            TextExampleRecord(
                example_id="biobio_0", text="She performed a triple bypass surgery.",
                documented_professions=["surgeon"],
            )
        ]

        with patch("anthropic.Anthropic", return_value=mock_client):
            judgments = verify_text_feature(
                claimed_concept="surgeon",
                example_records=records,
                n_repeats=3,
                api_key="fake-key-for-test",
            )

        assert len(judgments) == 3
        assert all(j.status == "confirmed" for j in judgments)
        assert mock_client.messages.create.call_count == 3


# ── Consistency rate ──────────────────────────────────────────────────────────

class TestConsistencyRate:
    def _judgment(self, status):
        from biolens.eval.auto_verify import VerifierJudgment

        return VerifierJudgment(status=status, confidence="high", reasoning="x")

    def test_all_agree_is_1(self):
        from biolens.eval.auto_verify import consistency_rate

        judgments = [self._judgment("confirmed") for _ in range(5)]
        assert consistency_rate(judgments) == pytest.approx(1.0)

    def test_partial_agreement(self):
        from biolens.eval.auto_verify import consistency_rate

        judgments = [
            self._judgment("confirmed"), self._judgment("confirmed"),
            self._judgment("confirmed"), self._judgment("spurious"),
        ]
        assert consistency_rate(judgments) == pytest.approx(0.75)

    def test_empty_list_is_zero(self):
        from biolens.eval.auto_verify import consistency_rate

        assert consistency_rate([]) == 0.0

    def test_single_judgment_is_trivially_consistent(self):
        from biolens.eval.auto_verify import consistency_rate

        assert consistency_rate([self._judgment("plausible")]) == pytest.approx(1.0)
