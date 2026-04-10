from __future__ import annotations

import unittest

from scripts.quality import compatibility_contract_guard, release_metadata_guard


class ScriptsSmokeTestCase(unittest.TestCase):
    def test_import_release_metadata_guard_sem_erro(self) -> None:
        from scripts.quality import release_metadata_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_compatibility_contract_guard_sem_erro(self) -> None:
        from scripts.quality import compatibility_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_extractors_identificam_versoes_basicas(self) -> None:
        footer = 'versao: str = "v73"'
        changelog = '## v73 — 2026-04-10\n\nResumo:'
        baseline = 'A baseline oficial vigente desta base é **v73**.'

        self.assertEqual(release_metadata_guard.extract_footer_version(footer), 'v73')
        self.assertEqual(release_metadata_guard.extract_changelog_version(changelog), 'v73')
        self.assertEqual(release_metadata_guard.extract_baseline_version(baseline), 'v73')

    def test_release_metadata_guard_passa_na_base_sincronizada(self) -> None:
        self.assertEqual(release_metadata_guard.main(), 0)

    def test_compatibility_contract_guard_passa_na_base_sincronizada(self) -> None:
        self.assertEqual(compatibility_contract_guard.main(), 0)
