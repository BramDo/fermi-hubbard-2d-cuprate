"""Explicit offline control for interpreting archived t-only results.

No CLI, export, backend or submission entry point is provided here. New
experiments must use one of the active circuit families instead.
"""
from qiskit import QuantumCircuit

from cuprate_2d_hubbard_pipeline import Cuprate2DConfig
from cuprate_2d_qiskit_circuit import (
    add_number_preserving_hopping_ansatz, add_onsite_terms, prepare_initial_state,
)


def build_historical_t_only(cfg: Cuprate2DConfig, *, measured: bool = False) -> QuantumCircuit:
    circuit = QuantumCircuit(cfg.qubits)
    prepare_initial_state(circuit, cfg)
    for _ in range(cfg.steps):
        add_onsite_terms(circuit, cfg)
        add_number_preserving_hopping_ansatz(circuit, cfg, include_tprime=False)
    circuit.metadata = {'circuit_family': 'number_preserving_t_only',
                        'historical_offline_control': True, 'active_hardware_route': False}
    if measured:
        circuit.measure_all()
    return circuit
