from __future__ import annotations

import contextlib
import io
import unittest

from scripts.quality import canonical_paths_reference_guard, checks_registry_consumers_guard, checks_registry_contract_guard, checks_registry_schema_guard, compatibility_contract_guard, documentation_commands_examples_guard, governance_docs_crosslinks_guard, operational_checks_contract_guard, protected_scope_hash_guard, quality_gate_composition_guard, quality_runtime_budget_guard, release_artifacts_hygiene_guard, release_manifest_guard, release_metadata_guard, runtime_dependencies_contract_guard, script_cli_contract_guard, script_exit_codes_contract_guard
from scripts.reports import maintenance_command_journal, maintenance_handoff_pack, maintenance_refresh_bundle, maintenance_reports_cleanup, maintenance_reports_index, maintenance_resume_brief, maintenance_snapshot_report, release_health_report


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

    def test_import_maintenance_command_journal_sem_erro(self) -> None:
        from scripts.reports import maintenance_command_journal as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_handoff_pack_sem_erro(self) -> None:
        from scripts.reports import maintenance_handoff_pack as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_resume_brief_sem_erro(self) -> None:
        from scripts.reports import maintenance_resume_brief as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_reports_cleanup_sem_erro(self) -> None:
        from scripts.reports import maintenance_reports_cleanup as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_refresh_bundle_sem_erro(self) -> None:
        from scripts.reports import maintenance_refresh_bundle as imported  # noqa: F401
        self.assertTrue(hasattr(imported, 'main'))

    def test_import_maintenance_reports_index_sem_erro(self) -> None:
        from scripts.reports import maintenance_reports_index as imported  # noqa: F401
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

        report = maintenance_snapshot_report.build_report('v93', datetime(2026, 4, 11, 17, 0, 0))
        self.assertIn('MAINTENANCE_SNAPSHOT_REPORT — v93', report)
        self.assertIn('Checks oficiais cadastrados', report)
        self.assertIn('Compatibilidade temporária', report)

    def test_maintenance_command_journal_builders_tem_titulos(self) -> None:
        from datetime import datetime

        report = maintenance_command_journal.build_markdown('v95', datetime(2026, 4, 11, 18, 0, 0))
        plain = maintenance_command_journal.build_plain_text('v95', datetime(2026, 4, 11, 18, 0, 0))
        self.assertIn('MAINTENANCE_COMMAND_JOURNAL — v95', report)
        self.assertIn('Ordem prática sugerida', report)
        self.assertIn('MAINTENANCE_COMMAND_JOURNAL — v95', plain)
        self.assertIn('Cenários de uso', plain)

    def test_maintenance_refresh_bundle_console_lines_tem_titulos(self) -> None:
        summary = {
            'version': 'v103',
            'steps': [
                maintenance_refresh_bundle.RefreshStep(name='maintenance_snapshot_report', outputs=[]),
                maintenance_refresh_bundle.RefreshStep(name='maintenance_handoff_pack', outputs=[]),
                maintenance_refresh_bundle.RefreshStep(name='maintenance_reports_index', outputs=[]),
            ],
            'outputs': [],
            'cleanup_suggestion': 'python scripts/reports/maintenance_reports_cleanup.py --dry-run',
        }
        lines = maintenance_refresh_bundle.build_console_lines(summary)
        joined = '\n'.join(lines)
        self.assertIn('MAINTENANCE REFRESH BUNDLE', joined)
        self.assertIn('maintenance_snapshot_report', joined)
        self.assertIn('maintenance_handoff_pack', joined)
        self.assertIn('maintenance_reports_index', joined)

    def test_maintenance_reports_index_build_report_tem_titulos(self) -> None:
        from datetime import datetime

        report = maintenance_reports_index.build_report('v103', datetime(2026, 4, 11, 21, 30, 0))
        self.assertIn('MAINTENANCE_REPORTS_INDEX — v103', report)
        self.assertIn('Artefatos operacionais mais recentes', report)
        self.assertIn('Comandos relacionados', report)

    def test_maintenance_handoff_pack_gera_zip(self) -> None:
        output_path = maintenance_handoff_pack.build_handoff_pack()
        try:
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.suffix, '.zip')
            self.assertIn('maintenance_handoff_', output_path.name)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_maintenance_resume_brief_builders_tem_titulos(self) -> None:
        from datetime import datetime

        report = maintenance_resume_brief.build_markdown('v93', datetime(2026, 4, 11, 17, 30, 0))
        plain = maintenance_resume_brief.build_plain_text('v93', datetime(2026, 4, 11, 17, 30, 0))
        self.assertIn('MAINTENANCE_RESUME_BRIEF — v93', report)
        self.assertIn('Escopo congelado', report)
        self.assertIn('MAINTENANCE_RESUME_BRIEF — v93', plain)
        self.assertIn('Prompt de continuidade', plain)

    def test_maintenance_command_journal_gera_markdown_e_texto(self) -> None:
        md_path, txt_path = maintenance_command_journal.write_command_journal()
        try:
            self.assertTrue(md_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertEqual(md_path.suffix, ' .md'.strip())
            self.assertEqual(txt_path.suffix, ' .txt'.strip())
            self.assertIn('MAINTENANCE_COMMAND_JOURNAL', md_path.read_text(encoding='utf-8'))
            self.assertIn('Ordem prática sugerida', txt_path.read_text(encoding='utf-8'))
        finally:
            if md_path.exists():
                md_path.unlink()
            if txt_path.exists():
                txt_path.unlink()

    def test_maintenance_resume_brief_gera_markdown_e_texto(self) -> None:
        md_path, txt_path = maintenance_resume_brief.write_resume_brief()
        try:
            self.assertTrue(md_path.exists())
            self.assertTrue(txt_path.exists())
            self.assertEqual(md_path.suffix, '.md')
            self.assertEqual(txt_path.suffix, '.txt')
            self.assertIn('MAINTENANCE_RESUME_BRIEF', md_path.read_text(encoding='utf-8'))
            self.assertIn('Prompt de continuidade', txt_path.read_text(encoding='utf-8'))
        finally:
            if md_path.exists():
                md_path.unlink()
            if txt_path.exists():
                txt_path.unlink()

    def test_maintenance_reports_cleanup_remove_artefatos_conhecidos(self) -> None:
        md_path, txt_path = maintenance_resume_brief.write_resume_brief()
        journal_md, journal_txt = maintenance_command_journal.write_command_journal()
        index_path = maintenance_reports_index.write_reports_index()
        summary = maintenance_reports_cleanup.cleanup_reports()
        try:
            self.assertTrue(bool(summary["clean_after"]))
            self.assertEqual(sorted(path.name for path in maintenance_reports_cleanup.OUTPUT_DIR.iterdir()), [".gitkeep"])
            archive_path = summary["archive_path"]
            self.assertIsNotNone(archive_path)
            archive_dir = archive_path
            self.assertTrue(archive_dir.exists())
            self.assertTrue((archive_dir / md_path.name).exists())
            self.assertTrue((archive_dir / txt_path.name).exists())
            self.assertTrue((archive_dir / journal_md.name).exists())
            self.assertTrue((archive_dir / journal_txt.name).exists())
            self.assertTrue((archive_dir / index_path.name).exists())
        finally:
            archive_path = summary.get("archive_path")
            if archive_path is not None and archive_path.exists():
                import shutil

                shutil.rmtree(archive_path)
            for path in (md_path, txt_path, journal_md, journal_txt, index_path):
                if path.exists():
                    path.unlink()

    def test_maintenance_reports_cleanup_preserva_inesperado_por_padrao(self) -> None:
        unexpected = maintenance_reports_cleanup.OUTPUT_DIR / "unexpected_debug_note.txt"
        unexpected.write_text("debug", encoding="utf-8")
        try:
            summary = maintenance_reports_cleanup.cleanup_reports()
            self.assertFalse(bool(summary["clean_after"]))
            self.assertTrue(unexpected.exists())
        finally:
            if unexpected.exists():
                unexpected.unlink()

    def test_maintenance_reports_index_gera_markdown(self) -> None:
        output_path = maintenance_reports_index.write_reports_index()
        try:
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, 'MAINTENANCE_REPORTS_INDEX.md')
            content = output_path.read_text(encoding='utf-8')
            self.assertIn('MAINTENANCE_REPORTS_INDEX', content)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_maintenance_refresh_bundle_gera_artefatos(self) -> None:
        summary = maintenance_refresh_bundle.refresh_bundle()
        outputs = list(summary['outputs'])
        try:
            self.assertGreaterEqual(len(outputs), 7)
            self.assertTrue(any(path.name.startswith('maintenance_snapshot_') and path.suffix == '.md' for path in outputs))
            self.assertTrue(any(path.name.startswith('maintenance_resume_brief_') and path.suffix == '.md' for path in outputs))
            self.assertTrue(any(path.name.startswith('maintenance_resume_brief_') and path.suffix == '.txt' for path in outputs))
            self.assertTrue(any(path.name.startswith('maintenance_command_journal_') and path.suffix == '.md' for path in outputs))
            self.assertTrue(any(path.name.startswith('maintenance_command_journal_') and path.suffix == '.txt' for path in outputs))
            self.assertTrue(any(path.name.startswith('maintenance_handoff_') and path.suffix == '.zip' for path in outputs))
            self.assertTrue(any(path.name == 'MAINTENANCE_REPORTS_INDEX.md' for path in outputs))
        finally:
            for path in outputs:
                if path.exists():
                    path.unlink()

    def test_maintenance_snapshot_report_gera_markdown(self) -> None:
        output_path = maintenance_snapshot_report.write_snapshot_report()
        try:
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.suffix, '.md')
            content = output_path.read_text(encoding='utf-8')
            self.assertIn('MAINTENANCE_SNAPSHOT_REPORT', content)
        finally:
            if output_path.exists():
                output_path.unlink()
