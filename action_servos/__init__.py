"""PCA9685 servo control: robotic arm (2 DOF) + head pan/tilt (2 DOF)."""

from action_servos.config import (
    DEFAULT_I2C_BUS,
    DEFAULT_MAX_US,
    DEFAULT_MIN_US,
    DEFAULT_PCA9685_ADDRESS,
    DEFAULT_PWM_FREQUENCY_HZ,
    JointSpec,
    ServoLayout,
)
from action_servos.groups import (
    ArmController,
    HeadController,
    ServoOrchestrator,
    clamp_pulse,
    normalized_to_us,
    presets_head_pose,
    us_to_normalized,
)
from action_servos.hardware import PCA9685

__all__ = [
    "ArmController",
    "DEFAULT_I2C_BUS",
    "DEFAULT_MAX_US",
    "DEFAULT_MIN_US",
    "DEFAULT_PCA9685_ADDRESS",
    "DEFAULT_PWM_FREQUENCY_HZ",
    "HeadController",
    "JointSpec",
    "PCA9685",
    "ServoLayout",
    "ServoOrchestrator",
    "clamp_pulse",
    "normalized_to_us",
    "presets_head_pose",
    "us_to_normalized",
]
