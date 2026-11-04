"""Regressões da issue #25, sem tocar na lógica do sorteio."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from unittest import TestCase, TestLoader
from unittest.mock import patch

from scripts.quality import protected_scope_hash_guard, release_metadata_guard


ROOT = Path(__file__).resolve().parents[1]


def _flatten(suite):
    for item in suite:
        if hasattr(item, "_testMethodName"):
            yield item
        else:
            yield from _flatten(item)


class GovernanceIssue25Tests(TestCase):
    def test_extractors_legados_e_fallback_sem_git(self):
        self.assertEqual(
            release_metadata_guard.extract_footer_version('versao: str = "v88"'),
            "v88",
        )
        self.assertEqual(
            release_metadata_guard.extract_footer_version(
                'APP_VERSION_FALLBACK = "v129"'
            ),
            "v129",
        )
        self.assertEqual(
            release_metadata_guard.extract_changelog_version(
                "## v129 — 2026-06-16\n\n## v128 — 2026-06-16"
            ),
            "v129",
        )
        self.assertEqual(
            release_metadata_guard.extract_baseline_version(
                "A baseline oficial vigente desta base é **v129**."
            ),
            "v129",
        )
        self.assertIsNone(release_metadata_guard.extract_footer_version("sem versao"))
        self.assertIsNone(release_metadata_guard.extract_changelog_version("# Geral"))
        self.assertIsNone(release_metadata_guard.extract_baseline_version("v129"))

    def test_manifesto_protegido_correponde_aos_bytes_da_main(self):
        path = ROOT / "docs/releases/PROTECTED_SCOPE_HASHES.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["baseline_version"], "v127")
        self.assertEqual(
            set(data["protected_files"]), {"app.py", "ui/review_view.py"}
        )
        self.assertEqual(
            data["reconciled_commit"],
            "83deb1646367bae1f738c9ae7f386cb2ccf87408",
        )
        for rel_path, spec in data["protected_files"].items():
            with self.subTest(path=rel_path):
                expected = hashlib.sha256((ROOT / rel_path).read_bytes()).hexdigest()
                self.assertEqual(spec["sha256"], expected)
                self.assertEqual(spec["status"], "frozen")
                self.assertNotEqual(spec["previous_sha256"], expected)

    def test_guard_de_hash_continua_detectando_alteracoes(self):
        output = io.StringIO()
        with patch.object(
            protected_scope_hash_guard,
            "sha256_file",
            return_value="0" * 64,
        ), contextlib.redirect_stdout(output):
            code = protected_scope_hash_guard.main()

        self.assertEqual(code, 1)
        self.assertIn("Hash divergente em arquivo protegido", output.getvalue())

    def test_descoberta_nao_duplica_suites_e_alias_legado_funciona(self):
        loader = TestLoader()
        suite = loader.discover(
            str(ROOT / "tests"),
            pattern="test_*.py",
            top_level_dir=str(ROOT),
        )
        ids = [test.id() for test in _flatten(suite)]
        duplicados = {name: count for name, count in Counter(ids).items() if count > 1}
        self.assertFalse(duplicados, f"Testes descobertos repetidamente: {duplicados}")
        self.assertGreater(len(ids), 50)
        legacy = TestLoader().loadTestsFromName("tests.test_smoke_base")
        self.assertGreater(legacy.countTestCases(), 40)


if __name__ == "__main__":
    import unittest

    unittest.main()
