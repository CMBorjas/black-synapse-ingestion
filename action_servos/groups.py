"""High-level arm (2 DOF) and head (pan/tilt) on a shared PCA9685."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from action_servos.config import JointSpec, ServoLayout
from action_servos.hardware import PCA9685

logger = logging.getLogger(__name__)


def clamp_pulse(spec: JointSpec, pulse_us: float) -> float:
    return max(spec.min_us, min(spec.max_us, float(pulse_us)))


def normalized_to_us(spec: JointSpec, n: float) -> float:
    """Map n in [-1, 1] to [min_us, max_us] linearly."""
    n = max(-1.0, min(1.0, float(n)))
    mid = (spec.min_us + spec.max_us) / 2.0
    half = (spec.max_us - spec.min_us) / 2.0
    return mid + n * half


def us_to_normalized(spec: JointSpec, pulse_us: Optional[float]) -> float:
    """Inverse of normalized_to_us; missing pulse defaults to 0.0 (center of range)."""
    if pulse_us is None:
        return 0.0
    mid = (spec.min_us + spec.max_us) / 2.0
    half = (spec.max_us - spec.min_us) / 2.0
    if half <= 0:
        return 0.0
    n = (float(pulse_us) - mid) / half
    return max(-1.0, min(1.0, n))


@dataclass
class ArmState:
    j0: Optional[float] = None
    j1: Optional[float] = None


@dataclass
class HeadState:
    pan: Optional[float] = None
    tilt: Optional[float] = None


class ArmController:
    def __init__(self, pca: PCA9685, j0: JointSpec, j1: JointSpec) -> None:
        self._pca = pca
        self._j0 = j0
        self._j1 = j1
        self._state = ArmState()

    @property
    def joint0_spec(self) -> JointSpec:
        return self._j0

    @property
    def joint1_spec(self) -> JointSpec:
        return self._j1

    @property
    def last_pulses(self) -> Tuple[Optional[float], Optional[float]]:
        return (self._state.j0, self._state.j1)

    def set_pulses(self, j0_us: float, j1_us: float) -> None:
        u0 = clamp_pulse(self._j0, j0_us)
        u1 = clamp_pulse(self._j1, j1_us)
        self._pca.set_channel_pulse_us(self._j0.channel, u0)
        self._pca.set_channel_pulse_us(self._j1.channel, u1)
        self._state.j0, self._state.j1 = u0, u1
        logger.debug("arm pulses us: (%.1f, %.1f)", u0, u1)

    def set_normalized(self, j0: float, j1: float) -> None:
        self.set_pulses(normalized_to_us(self._j0, j0), normalized_to_us(self._j1, j1))

    def center(self) -> None:
        self.set_pulses(self._j0.center_us, self._j1.center_us)

    def move_ramp(
        self,
        j0_us: float,
        j1_us: float,
        duration_s: float = 0.4,
        steps: int = 20,
    ) -> None:
        t0 = clamp_pulse(self._j0, j0_us)
        t1 = clamp_pulse(self._j1, j1_us)
        if self._state.j0 is None or self._state.j1 is None:
            self.set_pulses(t0, t1)
            return
        start0, start1 = self._state.j0, self._state.j1
        duration_s = max(0.01, float(duration_s))
        steps = max(2, int(steps))
        for i in range(1, steps + 1):
            a = i / float(steps)
            self.set_pulses(
                start0 + (t0 - start0) * a,
                start1 + (t1 - start1) * a,
            )
            time.sleep(duration_s / steps)


class HeadController:
    def __init__(self, pca: PCA9685, pan: JointSpec, tilt: JointSpec) -> None:
        self._pca = pca
        self._pan = pan
        self._tilt = tilt
        self._state = HeadState()

    @property
    def pan_spec(self) -> JointSpec:
        return self._pan

    @property
    def tilt_spec(self) -> JointSpec:
        return self._tilt

    @property
    def last_pulses(self) -> Tuple[Optional[float], Optional[float]]:
        return (self._state.pan, self._state.tilt)

    def set_pulses(self, pan_us: float, tilt_us: float) -> None:
        pu = clamp_pulse(self._pan, pan_us)
        tu = clamp_pulse(self._tilt, tilt_us)
        self._pca.set_channel_pulse_us(self._pan.channel, pu)
        self._pca.set_channel_pulse_us(self._tilt.channel, tu)
        self._state.pan, self._state.tilt = pu, tu
        logger.debug("head pulses us: pan=%.1f tilt=%.1f", pu, tu)

    def set_normalized(self, pan: float, tilt: float) -> None:
        self.set_pulses(normalized_to_us(self._pan, pan), normalized_to_us(self._tilt, tilt))

    def center(self) -> None:
        self.set_pulses(self._pan.center_us, self._tilt.center_us)

    def move_ramp(
        self,
        pan_us: float,
        tilt_us: float,
        duration_s: float = 0.4,
        steps: int = 20,
    ) -> None:
        pt = clamp_pulse(self._pan, pan_us)
        tt = clamp_pulse(self._tilt, tilt_us)
        if self._state.pan is None or self._state.tilt is None:
            self.set_pulses(pt, tt)
            return
        sp, st = self._state.pan, self._state.tilt
        duration_s = max(0.01, float(duration_s))
        steps = max(2, int(steps))
        for i in range(1, steps + 1):
            a = i / float(steps)
            self.set_pulses(sp + (pt - sp) * a, st + (tt - st) * a)
            time.sleep(duration_s / steps)


class ServoOrchestrator:
    """Shared PCA9685 with arm + head; use for full-robot presets and estop."""

    def __init__(self, layout: Optional[ServoLayout] = None) -> None:
        self.layout = layout or ServoLayout.default_layout()
        self._pca: Optional[PCA9685] = None
        self._arm_ctl: Optional[ArmController] = None
        self._head_ctl: Optional[HeadController] = None

    def open(self, bus: int, address: int, frequency_hz: float = 50.0) -> None:
        self.close()
        self._pca = PCA9685(bus=bus, address=address, frequency_hz=frequency_hz)
        self._pca.open()
        L = self.layout
        self._arm_ctl = ArmController(self._pca, L.arm_joint0, L.arm_joint1)
        self._head_ctl = HeadController(self._pca, L.head_pan, L.head_tilt)

    def close(self) -> None:
        self._arm_ctl = None
        self._head_ctl = None
        if self._pca is not None:
            self._pca.close()
            self._pca = None

    def __enter__(self) -> ServoOrchestrator:
        from action_servos.config import (
            DEFAULT_I2C_BUS,
            DEFAULT_PCA9685_ADDRESS,
            DEFAULT_PWM_FREQUENCY_HZ,
        )

        self.open(DEFAULT_I2C_BUS, DEFAULT_PCA9685_ADDRESS, DEFAULT_PWM_FREQUENCY_HZ)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def pca(self) -> PCA9685:
        if self._pca is None:
            raise RuntimeError("ServoOrchestrator not opened; call open() or use a context manager")
        return self._pca

    @property
    def arm(self) -> ArmController:
        if self._arm_ctl is None:
            raise RuntimeError("ServoOrchestrator not opened; call open() or use a context manager")
        return self._arm_ctl

    @property
    def head(self) -> HeadController:
        if self._head_ctl is None:
            raise RuntimeError("ServoOrchestrator not opened; call open() or use a context manager")
        return self._head_ctl

    def all_center(self) -> None:
        self.arm.center()
        self.head.center()

    def estop_center(self) -> None:
        """Safe pose: all joints to calibrated center (no chip sleep)."""
        self.all_center()


def presets_head_pose(name: str) -> Optional[Tuple[float, float]]:
    """Named (pan, tilt) in normalized [-1, 1]. Extend as needed."""
    poses = {
        "neutral": (0.0, 0.0),
        "look_left": (-0.6, 0.0),
        "look_right": (0.6, 0.0),
        "look_up": (0.0, -0.5),
        "look_down": (0.0, 0.5),
    }
    return poses.get(name.lower())
