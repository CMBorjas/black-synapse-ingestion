"""Keyframe sequence playback across all servo joints."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from action_servos.groups import ServoOrchestrator, clamp_pulse, normalized_to_us


@dataclass
class Pose:
    """Target state for one keyframe. All fields are pulse µs; None = don't move that joint."""

    arm_j0: Optional[float] = None
    arm_j1: Optional[float] = None
    head_tilt: Optional[float] = None
    head_pan: Optional[float] = None
    ear: Optional[float] = None

    @classmethod
    def from_normalized(
        cls,
        orch: ServoOrchestrator,
        arm: Optional[tuple[float, float]] = None,
        head_tilt: Optional[float] = None,
        head_pan: Optional[float] = None,
        ear: Optional[float] = None,
    ) -> Pose:
        """Convenience constructor that accepts normalized [-1, 1] values."""
        L = orch.layout
        return cls(
            arm_j0=normalized_to_us(L.arm_joint0, arm[0]) if arm is not None else None,
            arm_j1=normalized_to_us(L.arm_joint1, arm[1]) if arm is not None else None,
            head_tilt=normalized_to_us(L.head_tilt, head_tilt) if head_tilt is not None else None,
            head_pan=normalized_to_us(L.head_pan, head_pan)
            if (head_pan is not None and L.head_pan is not None)
            else None,
            ear=normalized_to_us(L.ear, ear) if ear is not None else None,
        )


@dataclass
class Keyframe:
    """A target Pose plus the ramp duration to reach it from the previous keyframe."""

    pose: Pose
    duration_s: float = 0.5
    steps: int = 20


@dataclass
class Sequence:
    """Ordered list of keyframes; play() executes them blocking in order."""

    keyframes: List[Keyframe] = field(default_factory=list)

    def add(
        self,
        pose: Pose,
        duration_s: float = 0.5,
        steps: int = 20,
    ) -> Sequence:
        """Append a keyframe; returns self for chaining."""
        self.keyframes.append(Keyframe(pose=pose, duration_s=duration_s, steps=steps))
        return self

    def play(self, orch: ServoOrchestrator) -> None:
        """Execute all keyframes in order. Blocking."""
        for kf in self.keyframes:
            _ramp_to_pose(orch, kf.pose, kf.duration_s, kf.steps)


def _ramp_to_pose(
    orch: ServoOrchestrator,
    pose: Pose,
    duration_s: float,
    steps: int,
) -> None:
    """Ramp all specified joints simultaneously in a single timed loop."""
    L = orch.layout

    # Gather current positions, falling back to center if joint has no prior state.
    arm_lp = orch.arm.last_pulses
    s_arm0 = arm_lp[0] if arm_lp[0] is not None else L.arm_joint0.center_us
    s_arm1 = arm_lp[1] if arm_lp[1] is not None else L.arm_joint1.center_us

    head_lp = orch.head.last_pulses
    s_tilt = head_lp[1] if head_lp[1] is not None else L.head_tilt.center_us
    s_pan = (
        head_lp[0]
        if head_lp[0] is not None
        else (L.head_pan.center_us if L.head_pan is not None else 0.0)
    )

    ear_lp = orch.ear.last_pulse
    s_ear = ear_lp if ear_lp is not None else L.ear.center_us

    # Resolve targets; joints not mentioned in the pose stay at their current position.
    t_arm0 = clamp_pulse(L.arm_joint0, pose.arm_j0) if pose.arm_j0 is not None else s_arm0
    t_arm1 = clamp_pulse(L.arm_joint1, pose.arm_j1) if pose.arm_j1 is not None else s_arm1
    t_tilt = clamp_pulse(L.head_tilt, pose.head_tilt) if pose.head_tilt is not None else s_tilt
    t_pan = (
        clamp_pulse(L.head_pan, pose.head_pan)
        if (pose.head_pan is not None and L.head_pan is not None)
        else s_pan
    )
    t_ear = clamp_pulse(L.ear, pose.ear) if pose.ear is not None else s_ear

    move_arm = pose.arm_j0 is not None or pose.arm_j1 is not None
    move_head = pose.head_tilt is not None or pose.head_pan is not None
    move_ear = pose.ear is not None

    duration_s = max(0.01, float(duration_s))
    steps = max(2, int(steps))

    for i in range(1, steps + 1):
        t = i / float(steps)
        a = t * t * (3.0 - 2.0 * t)  # smoothstep: ease-in and ease-out
        if move_arm:
            orch.arm.set_pulses(
                s_arm0 + (t_arm0 - s_arm0) * a,
                s_arm1 + (t_arm1 - s_arm1) * a,
            )
        if move_head:
            orch.head.set_pulses(
                s_pan + (t_pan - s_pan) * a,
                s_tilt + (t_tilt - s_tilt) * a,
            )
        if move_ear:
            orch.ear.set_pulse(s_ear + (t_ear - s_ear) * a)
        time.sleep(duration_s / steps)
