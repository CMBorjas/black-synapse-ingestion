"""Pre-configured arm actions exposed to the LLM. No raw servo values leak out."""

from __future__ import annotations

from action_servos.groups import ServoOrchestrator, normalized_to_us
from action_servos.sequences import Pose, Sequence

# ---------------------------------------------------------------------------
# Calibration constants — tune these for the physical robot.
# These are the ONLY place raw servo geometry lives in the codebase.
# All values are normalized (-1..1): -1 = min pulse, 0 = center, 1 = max pulse.
# ---------------------------------------------------------------------------
_EXTEND_J0  =  0.5   # shoulder when fully extended
_EXTEND_J1  =  0.5   # elbow when fully extended
_RETRACT_J0 = -0.5   # shoulder when fully retracted
_RETRACT_J1 = -1     # elbow when fully retracted
_REST_J0    = -0.3   # shoulder at resting position
_REST_J1    = -0.5   # elbow at resting position
_POINT_J0   =  0.6   # shoulder for pointing gesture
_POINT_J1   =  0.2   # elbow for pointing gesture


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _extend_pose(orch: ServoOrchestrator, amount: int) -> Pose:
    """Build a Pose for extending the arm by `amount` percent (0=neutral, 100=full)."""
    t = max(0.0, min(1.0, amount / 100.0))
    L = orch.layout
    return Pose(
        arm_j0=normalized_to_us(L.arm_joint0, _lerp(0.0, _EXTEND_J0, t)),
        arm_j1=normalized_to_us(L.arm_joint1, _lerp(0.0, _EXTEND_J1, t)),
    )


def _retract_pose(orch: ServoOrchestrator, amount: int) -> Pose:
    """Build a Pose for retracting the arm by `amount` percent (0=neutral, 100=full)."""
    t = max(0.0, min(1.0, amount / 100.0))
    L = orch.layout
    return Pose(
        arm_j0=normalized_to_us(L.arm_joint0, _lerp(0.0, _RETRACT_J0, t)),
        arm_j1=normalized_to_us(L.arm_joint1, _lerp(0.0, _RETRACT_J1, t)),
    )


_SLOW_STEPS = 50  # interpolation points per keyframe — higher = smoother


def _wave_sequence(orch: ServoOrchestrator) -> Sequence:
    L = orch.layout

    def p(j0n: float, j1n: float) -> Pose:
        return Pose(
            arm_j0=normalized_to_us(L.arm_joint0, j0n),
            arm_j1=normalized_to_us(L.arm_joint1, j1n),
        )

    return (
        Sequence()
        .add(p(0.5, 0.3),  duration_s=2.5,  steps=_SLOW_STEPS)  # raise up
        .add(p(0.5, 0.7),  duration_s=2.5,  steps=_SLOW_STEPS)  # wave out
        .add(p(0.5, 0.1),  duration_s=2.5,  steps=_SLOW_STEPS)  # wave in
        .add(p(0.5, 0.7),  duration_s=2.5,  steps=_SLOW_STEPS)  # wave out
        .add(p(0.5, 0.1),  duration_s=2.5,  steps=_SLOW_STEPS)  # wave in
        .add(p(0.0, 0.0),  duration_s=2.5,  steps=_SLOW_STEPS)  # return to neutral
    )


def execute_action(orch: ServoOrchestrator, action: str, amount: int | None) -> str:
    """
    Dispatch a named action to the servo orchestrator.

    Returns a human-readable description of what happened.
    Raises ValueError for unknown actions.
    """
    action = action.strip().lower()

    if action == "extend":
        pct = amount if amount is not None else 100
        Sequence().add(_extend_pose(orch, pct), duration_s=2.5, steps=_SLOW_STEPS).play(orch)
        return f"Extended arm to {pct}%."

    elif action == "retract":
        pct = amount if amount is not None else 100
        Sequence().add(_retract_pose(orch, pct), duration_s=2.5, steps=_SLOW_STEPS).play(orch)
        return f"Retracted arm to {pct}%."

    elif action == "wave":
        _wave_sequence(orch).play(orch)
        return "Waved."

    elif action == "point":
        L = orch.layout
        pose = Pose(
            arm_j0=normalized_to_us(L.arm_joint0, _POINT_J0),
            arm_j1=normalized_to_us(L.arm_joint1, _POINT_J1),
        )
        Sequence().add(pose, duration_s=2.5, steps=_SLOW_STEPS).play(orch)
        return "Pointing."

    elif action == "rest":
        L = orch.layout
        pose = Pose(
            arm_j0=normalized_to_us(L.arm_joint0, _REST_J0),
            arm_j1=normalized_to_us(L.arm_joint1, _REST_J1),
        )
        Sequence().add(pose, duration_s=2.5, steps=_SLOW_STEPS).play(orch)
        return "Arm moved to rest position."

    elif action == "release":
        orch.arm.release()
        return "Arm torque released. Arm can be moved by hand."

    else:
        raise ValueError(
            f"Unknown action '{action}'. "
            "Valid actions: extend, retract, wave, point, rest, release."
        )
