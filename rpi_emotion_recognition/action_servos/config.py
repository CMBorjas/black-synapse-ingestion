"""Defaults for I2C, PCA9685, and per-joint channel maps (adjust to your wiring)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Typical Jetson header I2C bus; confirm with `i2cdetect -l` on device.
DEFAULT_I2C_BUS = 7
DEFAULT_PCA9685_ADDRESS = 0x40
DEFAULT_PWM_FREQUENCY_HZ = 50.0

# Standard hobby servo pulse range (microseconds); tune per servo in JointSpec.
DEFAULT_MIN_US = 900
DEFAULT_MAX_US = 2100


@dataclass(frozen=True)
class JointSpec:
    """Maps a logical joint to a PCA9685 channel and safe pulse limits."""

    channel: int
    min_us: float = DEFAULT_MIN_US
    max_us: float = DEFAULT_MAX_US
    center_us: float = 1500.0

    def __post_init__(self) -> None:
        if not 0 <= self.channel <= 15:
            raise ValueError(f"PCA9685 channel must be 0-15, got {self.channel}")
        if self.min_us >= self.max_us:
            raise ValueError("min_us must be < max_us")


@dataclass(frozen=True)
class ServoLayout:
    """Arm (shoulder + elbow), head tilt (up/down), ear; optional head pan if wired."""

    arm_joint0: JointSpec
    arm_joint1: JointSpec
    head_pan: Optional[JointSpec]
    head_tilt: JointSpec
    ear: JointSpec

    @classmethod
    def default_layout(cls) -> ServoLayout:
        """Channels: head up/down 0, ear 3, elbow 5, shoulder 6; no pan servo."""
        return cls(
            arm_joint0=JointSpec(6),
            arm_joint1=JointSpec(5),
            head_pan=None,
            head_tilt=JointSpec(8, min_us=1000, max_us=2000),
            ear=JointSpec(3),
        )
