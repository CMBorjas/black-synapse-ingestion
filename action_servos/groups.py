"""High-level arm, head (optional pan + tilt), and ear on a shared PCA9685."""

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
        self._released: bool = False

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

    def release(self) -> None:
        """Cut PWM signal to both arm joints; servos go limp."""
        self._pca.set_channel_full_off(self._j0.channel)
        self._pca.set_channel_full_off(self._j1.channel)
        self._released = True

    def resume(self) -> None:
        """Re-engage torque and restore last known position (or center if unknown)."""
        j0 = self._state.j0 if self._state.j0 is not None else self._j0.center_us
        j1 = self._state.j1 if self._state.j1 is not None else self._j1.center_us
        self.set_pulses(j0, j1)
        self._released = False

    def reset(self) -> None:
        """Wake chip, clear FULL_OFF state, and re-send last known arm position."""
        j0 = self._state.j0 if self._state.j0 is not None else self._j0.center_us
        j1 = self._state.j1 if self._state.j1 is not None else self._j1.center_us
        self._pca.reset_channel(self._j0.channel, j0)
        self._pca.reset_channel(self._j1.channel, j1)
        self._released = False

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
    def __init__(
        self,
        pca: PCA9685,
        tilt: JointSpec,
        pan: Optional[JointSpec] = None,
    ) -> None:
        self._pca = pca
        self._pan = pan
        self._tilt = tilt
        self._state = HeadState()
        self._released: bool = False

    @property
    def pan_spec(self) -> Optional[JointSpec]:
        return self._pan

    @property
    def tilt_spec(self) -> JointSpec:
        return self._tilt

    @property
    def last_pulses(self) -> Tuple[Optional[float], Optional[float]]:
        return (self._state.pan, self._state.tilt)

    def set_pulses(self, pan_us: float, tilt_us: float) -> None:
        tu = clamp_pulse(self._tilt, tilt_us)
        self._pca.set_channel_pulse_us(self._tilt.channel, tu)
        self._state.tilt = tu
        if self._pan is not None:
            pu = clamp_pulse(self._pan, pan_us)
            self._pca.set_channel_pulse_us(self._pan.channel, pu)
            self._state.pan = pu
            logger.debug("head pulses us: pan=%.1f tilt=%.1f", pu, tu)
        else:
            self._state.pan = None
            logger.debug("head pulses us: (no pan) tilt=%.1f", tu)

    def set_normalized(self, pan: float, tilt: float) -> None:
        pan_us = normalized_to_us(self._pan, pan) if self._pan is not None else 0.0
        self.set_pulses(pan_us, normalized_to_us(self._tilt, tilt))

    def center(self) -> None:
        pan_c = self._pan.center_us if self._pan is not None else 0.0
        self.set_pulses(pan_c, self._tilt.center_us)

    def release(self) -> None:
        """Cut PWM signal to head joints; servos go limp."""
        self._pca.set_channel_full_off(self._tilt.channel)
        if self._pan is not None:
            self._pca.set_channel_full_off(self._pan.channel)
        self._released = True

    def resume(self) -> None:
        """Re-engage torque and restore last known position (or center if unknown)."""
        tilt = self._state.tilt if self._state.tilt is not None else self._tilt.center_us
        pan = self._state.pan if self._state.pan is not None else (
            self._pan.center_us if self._pan is not None else 0.0
        )
        self.set_pulses(pan, tilt)
        self._released = False

    def reset(self) -> None:
        """Wake chip, clear FULL_OFF state, and re-send last known head position."""
        tilt = self._state.tilt if self._state.tilt is not None else self._tilt.center_us
        self._pca.reset_channel(self._tilt.channel, tilt)
        if self._pan is not None:
            pan = self._state.pan if self._state.pan is not None else self._pan.center_us
            self._pca.reset_channel(self._pan.channel, pan)
        self._released = False

    def move_ramp(
        self,
        pan_us: float,
        tilt_us: float,
        duration_s: float = 0.4,
        steps: int = 20,
    ) -> None:
        tt = clamp_pulse(self._tilt, tilt_us)
        pt = clamp_pulse(self._pan, pan_us) if self._pan is not None else 0.0
        if self._state.tilt is None or (self._pan is not None and self._state.pan is None):
            self.set_pulses(pt, tt)
            return
        st = self._state.tilt
        sp = self._state.pan if self._pan is not None else pt
        duration_s = max(0.01, float(duration_s))
        steps = max(2, int(steps))
        for i in range(1, steps + 1):
            a = i / float(steps)
            new_pan = sp + (pt - sp) * a if self._pan is not None else pt
            new_tilt = st + (tt - st) * a
            self.set_pulses(new_pan, new_tilt)
            time.sleep(duration_s / steps)


@dataclass
class EarState:
    pulse: Optional[float] = None


class EarController:
    def __init__(self, pca: PCA9685, spec: JointSpec) -> None:
        self._pca = pca
        self._spec = spec
        self._state = EarState()
        self._released: bool = False

    @property
    def spec(self) -> JointSpec:
        return self._spec

    @property
    def last_pulse(self) -> Optional[float]:
        return self._state.pulse

    def set_pulse(self, pulse_us: float) -> None:
        u = clamp_pulse(self._spec, pulse_us)
        self._pca.set_channel_pulse_us(self._spec.channel, u)
        self._state.pulse = u
        logger.debug("ear pulse us: %.1f", u)

    def set_normalized(self, n: float) -> None:
        self.set_pulse(normalized_to_us(self._spec, n))

    def center(self) -> None:
        self.set_pulse(self._spec.center_us)

    def release(self) -> None:
        """Cut PWM signal to ear servo; servo goes limp."""
        self._pca.set_channel_full_off(self._spec.channel)
        self._released = True

    def resume(self) -> None:
        """Re-engage torque and restore last known position (or center if unknown)."""
        pulse = self._state.pulse if self._state.pulse is not None else self._spec.center_us
        self.set_pulse(pulse)
        self._released = False

    def reset(self) -> None:
        """Wake chip, clear FULL_OFF state, and re-send last known ear position."""
        pulse = self._state.pulse if self._state.pulse is not None else self._spec.center_us
        self._pca.reset_channel(self._spec.channel, pulse)
        self._released = False

    def move_ramp(
        self,
        pulse_us: float,
        duration_s: float = 0.4,
        steps: int = 20,
    ) -> None:
        t = clamp_pulse(self._spec, pulse_us)
        if self._state.pulse is None:
            self.set_pulse(t)
            return
        start = self._state.pulse
        duration_s = max(0.01, float(duration_s))
        steps = max(2, int(steps))
        for i in range(1, steps + 1):
            a = i / float(steps)
            self.set_pulse(start + (t - start) * a)
            time.sleep(duration_s / steps)


class ServoOrchestrator:
    """Shared PCA9685 with arm + head (+ optional ear); use for full-robot presets and estop."""

    def __init__(self, layout: Optional[ServoLayout] = None) -> None:
        self.layout = layout or ServoLayout.default_layout()
        self._pca: Optional[PCA9685] = None
        self._arm_ctl: Optional[ArmController] = None
        self._head_ctl: Optional[HeadController] = None
        self._ear_ctl: Optional[EarController] = None

    def open(self, bus: int, address: int, frequency_hz: float = 50.0) -> None:
        self.close()
        self._pca = PCA9685(bus=bus, address=address, frequency_hz=frequency_hz)
        self._pca.open()
        L = self.layout
        self._arm_ctl = ArmController(self._pca, L.arm_joint0, L.arm_joint1)
        self._head_ctl = HeadController(self._pca, L.head_tilt, pan=L.head_pan)
        self._ear_ctl = EarController(self._pca, L.ear)

    def close(self) -> None:
        self._arm_ctl = None
        self._head_ctl = None
        self._ear_ctl = None
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

    @property
    def ear(self) -> EarController:
        if self._ear_ctl is None:
            raise RuntimeError("ServoOrchestrator not opened; call open() or use a context manager")
        return self._ear_ctl

    def all_center(self) -> None:
        self.arm.center()
        self.head.center()
        self.ear.center()

    def estop_center(self) -> None:
        """Safe pose: all joints to calibrated center (no chip sleep)."""
        self.all_center()

    def release_all(self) -> None:
        """Cut PWM to all joints; all servos go limp."""
        self.arm.release()
        self.head.release()
        self.ear.release()

    def resume_all(self) -> None:
        """Re-engage torque on all joints, restoring last known positions."""
        self.arm.resume()
        self.head.resume()
        self.ear.resume()

    def reset_all(self) -> None:
        """Wake chip and clear FULL_OFF state on all joints, restoring last known positions."""
        self.arm.reset()
        self.head.reset()
        self.ear.reset()


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
