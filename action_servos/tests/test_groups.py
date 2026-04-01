"""Pure logic tests (no I2C hardware)."""

import pytest

from action_servos.config import JointSpec, ServoLayout
from action_servos.groups import (
    clamp_pulse,
    normalized_to_us,
    presets_head_pose,
    us_to_normalized,
)


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
