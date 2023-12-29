"""Framework-wide constants."""

DEFAULT_SHOTS = 1000
"""Default number of shots."""

DEFAULT_REP_DELAY = 2.5e-4
"""Default delay between circuit executions."""

RESTLESS_REP_DELAY = 1.e-5
"""Delay between circuit executions in the restless mode."""

RZ_SIGN = -1.
"""Relative sign between the RZGate parameter and the angle of the resulting physical Rz gate.
Because RZGate(phi) is scheduled as ShiftPhase(-phi), this is ultimately the negated sign of the
frequency of the LO factor in the drive Hamiltonian (i.e. H_d = Re[e^{-RZ_SIGN * i omega t}) r(t)].
"""
