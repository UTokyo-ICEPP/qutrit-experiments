"""Framework-wide constants."""

DEFAULT_SHOTS = 1000
"""Default number of shots."""

DEFAULT_REP_DELAY = 2.5e-4
"""Default delay between circuit executions."""

RESTLESS_REP_DELAY = 2.e-5
"""Delay between circuit executions in the restless mode."""

RESTLESS_RUN_OPTIONS = {
    'rep_delay': RESTLESS_REP_DELAY,
    'init_qubits': False,
    'memory': True,
    'use_measure_esp': False
}
"""Full run options for restless experiments, copied from RestlessMixin."""

LO_SIGN = 1.
"""The sign of the frequency in the "local oscillator" factor e^{iωt} in the drive Hamiltonian
H_d = Re[e^{LO_SIGN * i omega t} r(t)] (b + bdag). Because RZGate(phi) is scheduled as
ShiftPhase(-phi), the physical angle of the resulting Rz gate is +phi when LO_SIGN==-1 and -phi when
LO_SIGN==+1."""

USE_CUSTOM_PULSES = False
"""Whether to allow use of custom pulses in calibrations. Custom pulses are required for full
functionalities, but a bug in the backend is making their use impossible for the time being (as of
Feb 20 2024)."""
