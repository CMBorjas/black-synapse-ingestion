"""Slow scripted motion to verify PCA9685 wiring without sudden jumps.

Run from repo root::

    python -m action_servos.slow_test

Uses small normalized deflections (about ±0.2). Stop with Ctrl+C; I2C is closed
in a finally block.
"""

from __future__ import annotations

import argparse
import logging
import sys

from action_servos.config import (
    DEFAULT_I2C_BUS,
    DEFAULT_PCA9685_ADDRESS,
    DEFAULT_PWM_FREQUENCY_HZ,
)
from action_servos.groups import ServoOrchestrator, normalized_to_us


def _arm_us(orch: ServoOrchestrator, n0: float, n1: float) -> tuple[float, float]:
    a = orch.arm
    return (
        normalized_to_us(a.joint0_spec, n0),
        normalized_to_us(a.joint1_spec, n1),
    )


def _head_tilt_us(orch: ServoOrchestrator, tilt_norm: float) -> tuple[float, float]:
    h = orch.head
    pan_us = h.pan_spec.center_us if h.pan_spec is not None else 0.0
    return (pan_us, normalized_to_us(h.tilt_spec, tilt_norm))


def _ear_us(orch: ServoOrchestrator, n: float) -> float:
    return normalized_to_us(orch.ear.spec, n)


def run_sequence(orch: ServoOrchestrator, duration_s: float, steps: int) -> None:
    d, s = duration_s, steps
    print("Centering (fast).")
    orch.all_center()

    print("Arm: slow move (shoulder/elbow slightly negative).")
    orch.arm.move_ramp(*_arm_us(orch, -0.2, -0.2), duration_s=d, steps=s)

    print("Arm: slow move (slight positive).")
    orch.arm.move_ramp(*_arm_us(orch, 0.15, 0.12), duration_s=d, steps=s)

    print("Head tilt: slow down, then up toward neutral.")
    pu, t_down = _head_tilt_us(orch, 0.35)
    orch.head.move_ramp(pu, t_down, duration_s=d, steps=s)
    pu2, t_up = _head_tilt_us(orch, -0.25)
    orch.head.move_ramp(pu2, t_up, duration_s=d, steps=s)

    print("Ear: slow out and back.")
    orch.ear.move_ramp(_ear_us(orch, 0.25), duration_s=d, steps=s)
    orch.ear.move_ramp(orch.ear.spec.center_us, duration_s=d, steps=s)

    print("Arm: slow return to center.")
    orch.arm.move_ramp(
        orch.arm.joint0_spec.center_us,
        orch.arm.joint1_spec.center_us,
        duration_s=d,
        steps=s,
    )

    print("Head tilt: slow center.")
    pc = orch.head.tilt_spec.center_us
    pp = orch.head.pan_spec.center_us if orch.head.pan_spec is not None else 0.0
    orch.head.move_ramp(pp, pc, duration_s=d, steps=s)

    print("Done (all joints near center).")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Slow hardware check: gentle ramps on arm, head tilt, ear",
    )
    p.add_argument("--bus", type=int, default=DEFAULT_I2C_BUS, help="I2C bus number")
    p.add_argument(
        "--address",
        type=lambda x: int(x, 0),
        default=DEFAULT_PCA9685_ADDRESS,
        help="PCA9685 address (e.g. 0x40)",
    )
    p.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_PWM_FREQUENCY_HZ,
        help="PWM frequency (Hz)",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Seconds per ramp segment (default: 10)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Interpolation steps per segment (default: 100)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    duration_s = max(0.5, float(args.duration))
    steps = max(10, int(args.steps))

    orch = ServoOrchestrator()
    try:
        orch.open(args.bus, args.address, args.hz)
    except OSError as e:
        print(
            f"I2C open failed (bus {args.bus} addr 0x{args.address:02x}): {e}",
            file=sys.stderr,
        )
        return 1

    try:
        print(
            f"Slow test: {duration_s:.1f}s per segment, {steps} steps. "
            "Press Ctrl+C to stop.\n",
        )
        run_sequence(orch, duration_s=duration_s, steps=steps)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted; centering and closing.", file=sys.stderr)
        try:
            orch.estop_center()
        except OSError:
            pass
        return 130
    finally:
        orch.close()


if __name__ == "__main__":
    raise SystemExit(main())
