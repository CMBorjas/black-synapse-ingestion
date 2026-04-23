"""Pure logic tests (no I2C hardware)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from action_servos.config import JointSpec, ServoLayout
from action_servos.groups import (
    ArmController,
    EarController,
    HeadController,
    clamp_pulse,
    normalized_to_us,
    presets_head_pose,
    us_to_normalized,
)
from action_servos.sequences import Keyframe, Pose, Sequence, _ramp_to_pose


def _mock_pca() -> Any:
    """Return a MagicMock that records set_channel_pulse_us and set_channel_full_off calls."""
    return MagicMock()


def _arm(pca: Any | None = None) -> ArmController:
    if pca is None:
        pca = _mock_pca()
    j0 = JointSpec(6, min_us=900.0, max_us=2100.0)
    j1 = JointSpec(5, min_us=900.0, max_us=2100.0)
    return ArmController(pca, j0, j1)


def _ear(pca: Any | None = None) -> EarController:
    if pca is None:
        pca = _mock_pca()
    return EarController(pca, JointSpec(3))


def test_clamp_pulse() -> None:
    s = JointSpec(0, min_us=1000.0, max_us=2000.0)
    assert clamp_pulse(s, 500) == 1000.0
    assert clamp_pulse(s, 2500) == 2000.0
    assert clamp_pulse(s, 1500) == 1500.0


def test_normalized_round_trip() -> None:
    s = JointSpec(0, min_us=1000.0, max_us=2000.0)
    assert normalized_to_us(s, -1.0) == 1000.0
    assert normalized_to_us(s, 1.0) == 2000.0
    assert normalized_to_us(s, 0.0) == 1500.0
    assert us_to_normalized(s, 1250.0) == pytest.approx(-0.5, abs=1e-6)


def test_us_to_normalized_none() -> None:
    s = JointSpec(0)
    assert us_to_normalized(s, None) == 0.0


def test_joint_spec_channel_bounds() -> None:
    with pytest.raises(ValueError):
        JointSpec(16)
    with pytest.raises(ValueError):
        JointSpec(0, min_us=2000, max_us=1000)


def test_presets_head_pose() -> None:
    assert presets_head_pose("neutral") is not None
    assert presets_head_pose("unknown_pose_xyz") is None


def test_servo_layout_default() -> None:
    L = ServoLayout.default_layout()
    assert L.arm_joint0.channel == 6
    assert L.arm_joint1.channel == 5
    assert L.head_pan is None
    assert L.head_tilt.channel == 0
    assert L.ear.channel == 3


# --- release / resume tests ---

def test_arm_release_calls_full_off() -> None:
    pca = _mock_pca()
    arm = _arm(pca)
    arm.release()
    assert arm._released is True
    pca.set_channel_full_off.assert_any_call(6)
    pca.set_channel_full_off.assert_any_call(5)
    assert pca.set_channel_full_off.call_count == 2


def test_arm_resume_restores_last_pulse() -> None:
    pca = _mock_pca()
    arm = _arm(pca)
    arm.set_pulses(1800.0, 1200.0)
    arm.release()
    arm.resume()
    assert arm._released is False
    assert arm.last_pulses == (1800.0, 1200.0)


def test_arm_resume_defaults_to_center() -> None:
    pca = _mock_pca()
    arm = _arm(pca)
    # No set_pulses called yet — state is None
    arm.resume()
    assert arm.last_pulses == (arm.joint0_spec.center_us, arm.joint1_spec.center_us)


def test_ear_release_and_resume() -> None:
    pca = _mock_pca()
    ear = _ear(pca)
    ear.set_pulse(1700.0)
    ear.release()
    assert ear._released is True
    pca.set_channel_full_off.assert_called_once_with(3)
    ear.resume()
    assert ear._released is False
    assert ear.last_pulse == 1700.0


def test_ear_resume_defaults_to_center() -> None:
    pca = _mock_pca()
    ear = _ear(pca)
    ear.resume()
    assert ear.last_pulse == ear.spec.center_us


# --- Pose / Sequence tests ---

def test_pose_from_normalized_arm() -> None:
    from action_servos.groups import ServoOrchestrator
    orch = ServoOrchestrator()
    # Patch layout directly without opening hardware
    L = ServoLayout.default_layout()
    orch.layout = L
    pose = Pose.from_normalized(orch, arm=(0.0, 0.0))
    assert pose.arm_j0 == pytest.approx(1500.0)
    assert pose.arm_j1 == pytest.approx(1500.0)
    assert pose.ear is None


def test_sequence_play_visits_poses_in_order() -> None:
    """Verify that play() ramps to each keyframe in order using a mock orchestrator."""
    from action_servos.groups import ServoOrchestrator

    visited: list[tuple[float, float]] = []

    pca = _mock_pca()
    arm = _arm(pca)

    # Intercept set_pulses to record targets reached at end of each ramp
    original_set_pulses = arm.set_pulses

    def recording_set_pulses(j0: float, j1: float) -> None:
        original_set_pulses(j0, j1)
        visited.append((j0, j1))

    arm.set_pulses = recording_set_pulses  # type: ignore[method-assign]

    orch = MagicMock(spec=ServoOrchestrator)
    orch.layout = ServoLayout.default_layout()
    orch.arm = arm
    orch.head.last_pulses = (None, None)
    orch.ear.last_pulse = None

    seq = (
        Sequence()
        .add(Pose(arm_j0=1800.0, arm_j1=1200.0), duration_s=0.01, steps=2)
        .add(Pose(arm_j0=1500.0, arm_j1=1500.0), duration_s=0.01, steps=2)
    )
    seq.play(orch)

    # Last visited value of each ramp should equal the keyframe target
    # (visited contains all intermediate + final steps; last two are final steps of each ramp)
    assert visited[-1] == pytest.approx((1500.0, 1500.0), abs=1.0)
    # The first ramp's final step should be close to (1800, 1200)
    # It's visited[1] (index 1 out of 0,1 for first ramp's 2 steps)
    assert visited[1] == pytest.approx((1800.0, 1200.0), abs=1.0)
