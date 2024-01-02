"""Framework-wide constants."""

DEFAULT_SHOTS = 1000
"""Default number of shots."""

DEFAULT_REP_DELAY = 2.5e-4
"""Default delay between circuit executions."""

RESTLESS_REP_DELAY = 1.e-5
"""Delay between circuit executions in the restless mode."""

LO_SIGN = 1.
"""The sign of the frequency in the "local oscillator" factor e^{iωt} in the drive Hamiltonian
H_d = Re[e^{LO_SIGN * i omega t} r(t)] (b + bdag). Because RZGate(phi) is scheduled as
ShiftPhase(-phi), the physical angle of the resulting Rz gate is +phi when LO_SIGN==-1 and -phi when
LO_SIGN==+1."""
