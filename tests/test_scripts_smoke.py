from __future__ import annotations

import contextlib
import io
import unittest

from scripts.quality import canonical_paths_reference_guard, compatibility_contract_guard, operational_checks_contract_guard, release_artifacts_hygiene_guard, release_metadata_guard, script_cli_contract_guard
from scripts.reports import release_health_report


class ScriptsSmokeTestCase(unittest.TestCase):
    def test_import_release_metadata_guard_sem_erro(self) -> None:
        from scripts.quality import release_metadata_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_compatibility_contract_guard_sem_erro(self) -> None:
        from scripts.quality import compatibility_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_operational_checks_contract_guard_sem_erro(self) -> None:
        from scripts.quality import operational_checks_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_canonical_paths_reference_guard_sem_erro(self) -> None:
        from scripts.quality import canonical_paths_reference_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_script_cli_contract_guard_sem_erro(self) -> None:
        from scripts.quality import script_cli_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_release_artifacts_hygiene_guard_sem_erro(self) -> None:
        from scripts.quality import release_artifacts_hygiene_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_release_health_report_sem_erro(self) -> None:
        from scripts.reports import release_health_report as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_extractors_identificam_versoes_basicas(self) -> None:
        footer = 'versao: str = "v78"'
        changelog = '## v78 — 2026-04-10\n\nResumo:'
        baseline = 'A baseline oficial vigente desta base é **v78**.'

        self.assertEqual(release_metadata_guard.extract_footer_version(footer), 'v78')
        self.assertEqual(release_metadata_guard.extract_changelog_version(changelog), 'v78')
        self.assertEqual(release_metadata_guard.extract_baseline_version(baseline), 'v78')

    def test_release_metadata_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(release_metadata_guard.main(), 0)

    def test_compatibility_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compatibility_contract_guard.main(), 0)

    def test_operational_checks_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(operational_checks_contract_guard.main(), 0)

    def test_canonical_paths_reference_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(canonical_paths_reference_guard.main(), 0)

    def test_script_cli_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(script_cli_contract_guard.main(), 0)

    def test_release_artifacts_hygiene_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(release_artifacts_hygiene_guard.main(), 0)

    def test_release_health_report_build_report_tem_titulos(self) -> None:
        from datetime import datetime

        report = release_health_report.build_report(
            'v78',
            datetime(2026, 4, 10, 15, 0, 0),
            [
                {
                    'name': 'quality_gate',
                    'command': 'python scripts/quality/quality_gate.py',
                    'returncode': 0,
                    'ok': True,
                    'excerpt': ['Quality gate concluído com sucesso.'],
                }
            ],
        )
        self.assertIn('RELEASE_HEALTH_REPORT — v78', report)
        self.assertIn('Inventário resumido de compatibilidade temporária', report)
        self.assertIn('quality_gate', report)

