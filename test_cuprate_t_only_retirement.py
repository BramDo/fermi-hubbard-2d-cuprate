"""Offline regression tests for retirement in the public historical 2D snapshot."""
import argparse
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from qiskit.quantum_info import Statevector

import cuprate_2d_fire_opal_runner as fire
import cuprate_2d_ibm_hardware_runner as ibm
import cuprate_2d_layout_scout as layout
import cuprate_2d_mps_circuit_baseline as mps
import cuprate_2d_qiskit_circuit as circuits
import cuprate_2d_shallow_circuit_scout as scout
from cuprate_2d_hubbard_pipeline import Cuprate2DConfig
from historical.cuprate_t_only import build_historical_t_only

ROOT = Path(__file__).resolve().parent
LEGACY = 'number_preserving_t_only'


class TOnlyRetirementTests(unittest.TestCase):
    def setUp(self):
        self.cfg = Cuprate2DConfig(lx=2, ly=2, holes=0, steps=2, dt=.05,
                                   u_over_t=8, t_prime_over_t=-.25)
        network = patch('socket.socket.connect', side_effect=AssertionError('Offline tests only'))
        network.start()
        self.addCleanup(network.stop)

    def parse(self, module, *extra):
        argv = ['test', '--config', 'unused.json', '--outdir', 'unused', *extra]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
            return module.parse_args()

    def test_public_families_remove_only_t_only(self):
        self.assertEqual(set(circuits.SUPPORTED_CIRCUIT_FAMILIES),
                         {'explicit_jw', 'number_preserving_t_tprime'})
        self.assertNotIn(LEGACY, scout.DEFAULT_FAMILIES)
        with self.assertRaisesRegex(ValueError, 'retired'):
            circuits.build_circuit(self.cfg, measured=False, circuit_family=LEGACY)

    def test_remaining_families_and_defaults_are_preserved(self):
        default, _ = circuits.build_circuit(self.cfg, measured=False)
        explicit, _ = circuits.build_circuit(self.cfg, measured=False, circuit_family='explicit_jw')
        self.assertEqual(default, explicit)
        for module in (circuits, ibm, fire, layout):
            self.assertEqual(self.parse(module).circuit_family, 'explicit_jw')
        for family in circuits.SUPPORTED_CIRCUIT_FAMILIES:
            with self.subTest(family=family):
                circuit, _ = circuits.build_circuit(self.cfg, measured=True, circuit_family=family)
                self.assertEqual((circuit.num_qubits, circuit.num_clbits), (8, 8))
                for module in (circuits, ibm, fire, layout, mps):
                    self.assertEqual(self.parse(module, '--circuit-family', family).circuit_family, family)

    def test_zero_tprime_is_still_a_valid_model_parameter(self):
        for family in circuits.SUPPORTED_CIRCUIT_FAMILIES:
            circuit, _ = circuits.build_circuit(replace(self.cfg, t_prime_over_t=0),
                                                measured=False, circuit_family=family)
            self.assertEqual(circuit.num_qubits, 8)

    def test_new_cli_actions_reject_retired_family(self):
        for module in (circuits, ibm, fire, layout, mps):
            with self.subTest(module=module.__name__), self.assertRaises(SystemExit):
                self.parse(module, '--circuit-family', LEGACY)
        for module in (ibm, fire):
            with self.assertRaises(SystemExit):
                self.parse(module, '--circuit-family', LEGACY, '--submit')

    def test_mps_requires_explicit_model_instead_of_silent_replacement(self):
        with self.assertRaises(SystemExit):
            self.parse(mps)

    def test_retrieval_accepts_legacy_but_never_submit_combination(self):
        for module in (ibm, fire):
            self.assertTrue(self.parse(module, '--circuit-family', LEGACY, '--retrieve').retrieve)
            for family in (LEGACY, 'explicit_jw'):
                with self.assertRaises(SystemExit):
                    self.parse(module, '--circuit-family', family, '--retrieve', '--submit')

    def test_unknown_family_is_rejected(self):
        for retrieve in (False, True):
            with self.assertRaises(ValueError):
                circuits.validate_circuit_family_action('not-a-family', retrieve=retrieve)

    def test_runner_refusal_precedes_config_and_provider_access(self):
        args = argparse.Namespace(circuit_family=LEGACY, retrieve=False, submit=True, outdir=Path('unused'))
        for module in (circuits, ibm, fire, layout):
            with self.subTest(module=module.__name__), \
                    patch.object(module, 'parse_args', return_value=args), \
                    patch.object(module, 'load_config') as config:
                with self.assertRaisesRegex(ValueError, 'retired'):
                    module.main()
                config.assert_not_called()

    def test_mps_refuses_before_config_and_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'must-not-exist'
            args = argparse.Namespace(circuit_family=LEGACY, outdir=output)
            with patch.object(mps, 'load_config') as config:
                with self.assertRaisesRegex(ValueError, 'retired'):
                    mps.run(args)
                config.assert_not_called()
            self.assertFalse(output.exists())

    def test_shallow_scout_rejects_before_backend(self):
        with self.assertRaisesRegex(ValueError, 'retired'):
            scout.build_shallow_candidate(self.cfg, LEGACY, measured=False)
        with patch.object(scout, 'parse_args', return_value=argparse.Namespace(family=[LEGACY])), \
                patch.object(scout, 'load_config') as config, \
                patch.object(scout, 'build_runtime_service') as service:
            with self.assertRaisesRegex(ValueError, 'retired'):
                scout.main()
            config.assert_not_called()
            service.assert_not_called()

    def test_direct_fire_opal_submission_refuses_before_credentials(self):
        with patch.object(fire, 'qctrl_api_key') as key, patch.object(fire, 'ibm_credentials') as credentials:
            with self.assertRaisesRegex(ValueError, 'retired'):
                fire.submit_fire_opal(Path('unused'), [], argparse.Namespace(circuit_family=LEGACY))
            key.assert_not_called()
            credentials.assert_not_called()

    def test_output_directory_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            marker = root/'submitted_jobs.json'
            for identity in ({'circuit_family': LEGACY}, {}, []):
                marker.write_text(json.dumps(identity))
                before = marker.read_bytes()
                with self.assertRaisesRegex(ValueError, 'new --outdir'):
                    circuits.ensure_output_circuit_family(root, 'explicit_jw')
                self.assertEqual(marker.read_bytes(), before)
            marker.write_text(json.dumps({'circuit_family': 'explicit_jw'}))
            circuits.ensure_output_circuit_family(root, 'explicit_jw')

    def test_ibm_retrieval_keeps_saved_identity_and_never_builds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/'config.json'
            config.write_text(json.dumps(asdict(self.cfg)))
            receipt = {'job_id': 'offline-saved-job', 'backend': 'offline-backend',
                       'circuit_family': LEGACY, 'roles': ['main']}
            (root/'submitted_jobs.json').write_text(json.dumps(receipt))
            argv = ['test', '--config', str(config), '--outdir', str(root),
                    '--circuit-family', LEGACY, '--retrieve']
            service = Mock()
            with patch.object(sys, 'argv', argv), patch.object(ibm, 'build_runtime_service', return_value=service), \
                    patch.object(ibm, 'wait_until_done', return_value='QUEUED'), \
                    patch.object(ibm, 'build_circuit') as build, patch.object(ibm, 'submit_job') as submit, \
                    contextlib.redirect_stdout(io.StringIO()):
                ibm.main()
                service.job.assert_called_once_with('offline-saved-job')
                build.assert_not_called()
                submit.assert_not_called()
            saved = json.loads((root/'submitted_jobs.json').read_text())
            self.assertEqual(saved['circuit_family'], LEGACY)
            self.assertEqual(saved['job_id'], receipt['job_id'])

    def test_fire_opal_retrieval_never_builds_or_submits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/'config.json'
            config.write_text(json.dumps(asdict(self.cfg)))
            argv = ['test', '--config', str(config), '--outdir', str(root),
                    '--circuit-family', LEGACY, '--retrieve']
            with patch.object(sys, 'argv', argv), \
                    patch.object(fire, 'retrieve_fire_opal', return_value=False) as retrieve, \
                    patch.object(fire, 'build_groups') as build, patch.object(fire, 'submit_fire_opal') as submit:
                fire.main()
                retrieve.assert_called_once()
                self.assertEqual(retrieve.call_args.args[2].circuit_family, LEGACY)
                build.assert_not_called()
                submit.assert_not_called()

    def test_historical_helper_preserves_one_step_scope(self):
        cfg = replace(self.cfg, steps=1, dt=.4)
        circuit = build_historical_t_only(cfg)
        self.assertTrue(circuit.metadata['historical_offline_control'])
        self.assertFalse(circuit.metadata['active_hardware_route'])
        expected = Statevector(circuit).probabilities()
        for case in (replace(cfg, u_over_t=0), replace(cfg, t_prime_over_t=0)):
            np.testing.assert_allclose(Statevector(build_historical_t_only(case)).probabilities(), expected, atol=1e-12)

    def test_public_launchers_stop_and_retrieval_wrappers_remain(self):
        guards, retrieves = 0, 0
        for path in (ROOT/'scripts').glob('*.ps1'):
            source = path.read_text(encoding='utf-8-sig')
            if LEGACY not in source:
                continue
            if path.name.startswith('retrieve_'):
                self.assertIn('--retrieve', source)
                self.assertNotIn('--submit', source)
                retrieves += 1
            else:
                lines = [line.strip() for line in source.splitlines()
                         if line.strip() and not line.lstrip().startswith('#')]
                self.assertEqual(len(lines), 1, path.name)
                self.assertTrue(lines[0].startswith('throw '), path.name)
                self.assertNotIn('wsl.exe', source)
                guards += 1
        self.assertEqual((guards, retrieves), (30, 12))

    def test_active_circuit_export_still_runs_locally(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root/'config.json'
            config.write_text(json.dumps(asdict(self.cfg)))
            output = root/'new-explicit'
            argv = ['test', '--config', str(config), '--outdir', str(output)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                circuits.main()
            summary = json.loads((output/'cuprate_2d_qiskit_circuit_summary.json').read_text())
            self.assertEqual(summary['circuit_family'], 'explicit_jw')
            self.assertEqual(summary['qubits'], 8)
            self.assertTrue((output/'cuprate_2d_explicit_jw_trotter.qpy').is_file())


if __name__ == '__main__':
    unittest.main()
