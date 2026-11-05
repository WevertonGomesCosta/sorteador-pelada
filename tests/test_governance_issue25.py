"""Regressões da issue #25, sem tocar na lógica do sorteio."""

from __future__ import annotations

import contextlib
import io
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, TestLoader
from unittest.mock import patch

from scripts.quality import (check_base, protected_scope_hash_guard, release_guard, release_metadata_guard, release_artifacts_hygiene_guard)


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

    def test_release_guard_reutiliza_o_contrato_atual_de_versao(self):
        self.assertEqual(release_guard.footer_version(), "v129")
        self.assertEqual(release_guard.latest_changelog_version(), "v129")
        self.assertEqual(
            release_guard.footer_version(),
            release_guard.latest_changelog_version(),
        )

    def test_changelog_contem_secoes_de_governanca_exigidas(self):
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for title in (
            "## Padrão oficial para novas entradas",
            "## Histórico técnico consolidado",
            "## Observações de congelamento vigentes",
        ):
            with self.subTest(title=title):
                self.assertIn(title, changelog)

    def test_check_base_normaliza_caminhos_no_windows(self):
        self.assertEqual(
            check_base.relative(check_base.ROOT / "ui" / "review_view.py"),
            "ui/review_view.py",
        )

    def test_hash_protegido_normaliza_apenas_finais_de_linha(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "codigo.py"
            path.write_bytes(b"print('ok')\n")
            esperado = protected_scope_hash_guard.sha256_file(path)
            path.write_bytes(b"print('ok')\r\n")
            self.assertEqual(protected_scope_hash_guard.sha256_file(path), esperado)
            path.write_bytes(b"print('outro')\r\n")
            self.assertNotEqual(protected_scope_hash_guard.sha256_file(path), esperado)

    def test_higiene_tolera_somente_pasta_vazia_bloqueada(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "ui" / "__pycache__"
            cache.mkdir(parents=True)
            with patch.object(Path, "rmdir", side_effect=PermissionError("WinError 5")):
                removed, compiled = release_artifacts_hygiene_guard.remove_bytecode_artifacts(root)
            self.assertEqual((removed, compiled), (0, 0))
            self.assertTrue(release_artifacts_hygiene_guard.is_empty_directory(cache))

            residual = cache / "residuo.pyc"
            residual.write_bytes(b"binary")
            self.assertFalse(release_artifacts_hygiene_guard.is_empty_directory(cache))

    def test_release_guard_detecta_cache_residual_e_tolera_vazio(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "ui" / "__pycache__"
            cache.mkdir(parents=True)
            with patch.object(
                release_guard.shutil,
                "rmtree",
                side_effect=PermissionError("WinError 5"),
            ):
                removed, compiled = release_guard.remove_bytecode_artifacts(root)
            self.assertEqual((removed, compiled), (0, 0))
            self.assertEqual(release_guard.find_artifacts(root), ([], []))

            (cache / "persistente.pyc").write_bytes(b"data")
            caches, bytecode = release_guard.find_artifacts(root)
            self.assertEqual(caches, [cache])
            self.assertEqual(len(bytecode), 1)

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
                expected = protected_scope_hash_guard.sha256_file(ROOT / rel_path)
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
