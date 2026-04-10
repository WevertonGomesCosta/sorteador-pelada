from __future__ import annotations

import unittest

from scripts.quality import compatibility_contract_guard, release_metadata_guard
from scripts.reports import release_health_report


class ScriptsSmokeTestCase(unittest.TestCase):
    def test_import_release_metadata_guard_sem_erro(self) -> None:
        from scripts.quality import release_metadata_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_compatibility_contract_guard_sem_erro(self) -> None:
        from scripts.quality import compatibility_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_release_health_report_sem_erro(self) -> None:
        from scripts.reports import release_health_report as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_extractors_identificam_versoes_basicas(self) -> None:
        footer = 'versao: str = "v74"'
        changelog = '## v74 — 2026-04-10\n\nResumo:'
        baseline = 'A baseline oficial vigente desta base é **v74**.'

        self.assertEqual(release_metadata_guard.extract_footer_version(footer), 'v74')
        self.assertEqual(release_metadata_guard.extract_changelog_version(changelog), 'v74')
        self.assertEqual(release_metadata_guard.extract_baseline_version(baseline), 'v74')

    def test_release_metadata_guard_passa_na_base_sincronizada(self) -> None:
        self.assertEqual(release_metadata_guard.main(), 0)

    def test_compatibility_contract_guard_passa_na_base_sincronizada(self) -> None:
        self.assertEqual(compatibility_contract_guard.main(), 0)

    def test_release_health_report_build_report_tem_titulos(self) -> None:
        from datetime import datetime

        report = release_health_report.build_report(
            'v74',
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
        self.assertIn('RELEASE_HEALTH_REPORT — v74', report)
        self.assertIn('Inventário resumido de compatibilidade temporária', report)
        self.assertIn('quality_gate', report)

