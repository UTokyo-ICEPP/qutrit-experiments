"""Framework-wide constants."""

DEFAULT_SHOTS = 1000
"""Default number of shots."""

DEFAULT_REP_DELAY = 2.5e-4
"""Default delay between circuit executions."""

RESTLESS_REP_DELAY = 1.e-5
"""Delay between circuit executions in the restless mode."""

RZ_SIGN = -1.
"""Qiskit convention: RZGate(phi) is scheduled as ShiftPhase(-phi), which effects a physical
Rz(-phi)."""
