from __future__ import annotations

import contextlib
import io
import unittest

from scripts.quality import canonical_paths_reference_guard, checks_registry_consumers_guard, checks_registry_contract_guard, checks_registry_schema_guard, compatibility_contract_guard, documentation_commands_examples_guard, governance_docs_crosslinks_guard, operational_checks_contract_guard, protected_scope_hash_guard, quality_gate_composition_guard, quality_runtime_budget_guard, release_artifacts_hygiene_guard, release_manifest_guard, release_metadata_guard, runtime_dependencies_contract_guard, script_cli_contract_guard, script_exit_codes_contract_guard
from scripts.reports import maintenance_snapshot_report, release_health_report


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

    def test_import_runtime_dependencies_contract_guard_sem_erro(self) -> None:
        from scripts.quality import runtime_dependencies_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_documentation_commands_examples_guard_sem_erro(self) -> None:
        from scripts.quality import documentation_commands_examples_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_release_manifest_guard_sem_erro(self) -> None:
        from scripts.quality import release_manifest_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_release_health_report_sem_erro(self) -> None:
        from scripts.reports import release_health_report as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_snapshot_report_sem_erro(self) -> None:
        from scripts.reports import maintenance_snapshot_report as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_script_exit_codes_contract_guard_sem_erro(self) -> None:
        from scripts.quality import script_exit_codes_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_governance_docs_crosslinks_guard_sem_erro(self) -> None:
        from scripts.quality import governance_docs_crosslinks_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_protected_scope_hash_guard_sem_erro(self) -> None:
        from scripts.quality import protected_scope_hash_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_checks_registry_contract_guard_sem_erro(self) -> None:
        from scripts.quality import checks_registry_contract_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_checks_registry_schema_guard_sem_erro(self) -> None:
        from scripts.quality import checks_registry_schema_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_checks_registry_consumers_guard_sem_erro(self) -> None:
        from scripts.quality import checks_registry_consumers_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_quality_gate_composition_guard_sem_erro(self) -> None:
        from scripts.quality import quality_gate_composition_guard as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_extractors_identificam_versoes_basicas(self) -> None:
        footer = 'versao: str = "v88"'
        changelog = '## v88 — 2026-04-11\n\nResumo:'
        baseline = 'A baseline oficial vigente desta base é **v88**.'

        self.assertEqual(release_metadata_guard.extract_footer_version(footer), 'v88')
        self.assertEqual(release_metadata_guard.extract_changelog_version(changelog), 'v88')
        self.assertEqual(release_metadata_guard.extract_baseline_version(baseline), 'v88')

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

    def test_runtime_dependencies_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime_dependencies_contract_guard.main(), 0)

    def test_documentation_commands_examples_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(documentation_commands_examples_guard.main(), 0)

    def test_release_manifest_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(release_manifest_guard.main(), 0)

    def test_quality_runtime_budget_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(quality_runtime_budget_guard.main(), 0)

    def test_script_exit_codes_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(script_exit_codes_contract_guard.main(), 0)

    def test_governance_docs_crosslinks_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(governance_docs_crosslinks_guard.main(), 0)

    def test_protected_scope_hash_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(protected_scope_hash_guard.main(), 0)

    def test_checks_registry_contract_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checks_registry_contract_guard.main(), 0)

    def test_checks_registry_schema_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checks_registry_schema_guard.main(), 0)

    def test_checks_registry_consumers_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(checks_registry_consumers_guard.main(), 0)

    def test_quality_gate_composition_guard_passa_na_base_sincronizada(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(quality_gate_composition_guard.main(), 0)

    def test_release_health_report_build_report_tem_titulos(self) -> None:
        from datetime import datetime

        report = release_health_report.build_report(
            'v88',
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
        self.assertIn('RELEASE_HEALTH_REPORT — v88', report)
        self.assertIn('Inventário resumido de compatibilidade temporária', report)
        self.assertIn('quality_gate', report)

    def test_maintenance_snapshot_report_build_report_tem_titulos(self) -> None:
        from datetime import datetime

        snapshot = maintenance_snapshot_report.build_snapshot(datetime(2026, 4, 11, 18, 0, 0))
        report = maintenance_snapshot_report.build_report(snapshot)
        self.assertIn('MAINTENANCE_SNAPSHOT_REPORT', report)
        self.assertIn('Compatibilidade temporária', report)
        self.assertIn('Escopo protegido', report)
