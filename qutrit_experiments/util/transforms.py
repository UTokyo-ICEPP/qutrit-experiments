"""Transformation utilities."""
from qiskit import pulse
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.qobj.converters.pulse_instruction import ParametricPulseShapes


def schedule_to_block(schedule: Schedule):
    """Convert a Schedule to ScheduleBlock. Am I sure this doesn't exist in Qiskit proper?"""
    # Convert to ScheduleBlock
    with pulse.build(name=schedule.name) as schedule_block:
        pulse.call(schedule)
    return schedule_block


def symbolic_pulse_to_waveform(schedule_block: ScheduleBlock):
    """Replace custom pulses to waveforms in place."""
    # .blocks returns a new container so we don't need to wrap the result further
    for block in schedule_block.blocks:
        if isinstance((inst := block), pulse.Play) and isinstance(inst.pulse, pulse.SymbolicPulse):
            # Check if the pulse is known by the backend
            try:
                ParametricPulseShapes(inst.pulse.pulse_type)
            except ValueError:
                waveform_inst = pulse.Play(inst.pulse.get_waveform(), inst.channel, name=inst.name)
                schedule_block.replace(inst, waveform_inst, inplace=True)
        elif isinstance(block, ScheduleBlock):
            symbolic_pulse_to_waveform(block)
