from __future__ import annotations

import unittest

from scripts.quality import release_metadata_guard


class ScriptsSmokeTestCase(unittest.TestCase):
    def test_import_release_metadata_guard_sem_erro(self) -> None:
        from scripts.quality import release_metadata_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_extractors_identificam_versoes_basicas(self) -> None:
        footer = 'versao: str = "v72"'
        changelog = '## v72 — 2026-04-10\n\nResumo:'
        baseline = 'A baseline oficial vigente desta base é **v72**.'

        self.assertEqual(release_metadata_guard.extract_footer_version(footer), 'v72')
        self.assertEqual(release_metadata_guard.extract_changelog_version(changelog), 'v72')
        self.assertEqual(release_metadata_guard.extract_baseline_version(baseline), 'v72')

    def test_release_metadata_guard_passa_na_base_sincronizada(self) -> None:
        self.assertEqual(release_metadata_guard.main(), 0)
