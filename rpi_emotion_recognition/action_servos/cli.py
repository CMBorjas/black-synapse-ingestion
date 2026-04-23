"""Manual jog on hardware: run from repo root as `python -m action_servos.cli`."""

from __future__ import annotations

import argparse
import logging
import sys

from action_servos.config import (
    DEFAULT_I2C_BUS,
    DEFAULT_PCA9685_ADDRESS,
    DEFAULT_PWM_FREQUENCY_HZ,
)
from action_servos.groups import ServoOrchestrator, presets_head_pose, us_to_normalized
from worker.app.arm_actions import execute_action


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jog PCA9685 arm + head servos")
    parser.add_argument("--bus", type=int, default=DEFAULT_I2C_BUS, help="I2C bus number")
    parser.add_argument(
        "--address",
        type=lambda x: int(x, 0),
        default=DEFAULT_PCA9685_ADDRESS,
        help="PCA9685 address (e.g. 0x40)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=DEFAULT_PWM_FREQUENCY_HZ,
        help="PWM frequency (Hz), typically 50 for hobby servos",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("center", help="Move arm, head tilt, and ear to center_us")

    p_arm = sub.add_parser("arm", help="Set arm joints (normalized -1..1 or pulse us)")
    p_arm.add_argument("--n0", type=float, default=None, help="joint0 normalized -1..1")
    p_arm.add_argument("--n1", type=float, default=None, help="joint1 normalized -1..1")
    p_arm.add_argument("--us0", type=float, default=None, help="joint0 pulse microseconds")
    p_arm.add_argument("--us1", type=float, default=None, help="joint1 pulse microseconds")

    p_head = sub.add_parser("head", help="Set head pan/tilt (normalized or us)")
    p_head.add_argument("--pan", type=float, default=None, help="pan -1..1")
    p_head.add_argument("--tilt", type=float, default=None, help="tilt -1..1")
    p_head.add_argument("--pan-us", type=float, default=None, dest="pan_us")
    p_head.add_argument("--tilt-us", type=float, default=None, dest="tilt_us")
    p_head.add_argument("--pose", type=str, default=None, help="named pose: neutral, look_left, ...")

    p_ear = sub.add_parser("ear", help="Set ear servo (normalized -1..1 or pulse us)")
    p_ear.add_argument("-n", type=float, default=None, dest="norm", help="normalized -1..1")
    p_ear.add_argument("--us", type=float, default=None, dest="us", help="pulse microseconds")

    p_seq = sub.add_parser("sequence", help="Run a named arm sequence (wave, point, rest, extend, retract)")
    p_seq.add_argument("name", type=str, help="sequence name: wave, point, rest, extend, retract")
    p_seq.add_argument("--amount", type=int, default=None, help="percent for extend/retract (0-100)")

    for _cmd, _help in (("release", "Cut PWM to joints (servos go limp)"),
                        ("resume", "Re-engage torque on joints (restores last position)"),
                        ("reset", "Wake chip + clear FULL_OFF state, re-send last position")):
        p = sub.add_parser(_cmd, help=_help)
        p.add_argument("--arm", action="store_true", default=False)
        p.add_argument("--head", action="store_true", default=False)
        p.add_argument("--ear", action="store_true", default=False)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        orch = ServoOrchestrator()
        orch.open(args.bus, args.address, args.hz)
    except OSError as e:
        print(f"I2C open failed (bus {args.bus} addr 0x{args.address:02x}): {e}", file=sys.stderr)
        return 1

    try:
        if args.cmd == "center":
            orch.all_center()
        elif args.cmd == "arm":
            arm = orch.arm
            if args.us0 is not None or args.us1 is not None:
                s0, s1 = arm.last_pulses
                if args.us0 is not None:
                    s0 = args.us0
                if args.us1 is not None:
                    s1 = args.us1
                if s0 is None:
                    s0 = arm.joint0_spec.center_us
                if s1 is None:
                    s1 = arm.joint1_spec.center_us
                arm.set_pulses(float(s0), float(s1))
            elif args.n0 is not None or args.n1 is not None:
                lp = arm.last_pulses
                n0 = args.n0 if args.n0 is not None else us_to_normalized(arm.joint0_spec, lp[0])
                n1 = args.n1 if args.n1 is not None else us_to_normalized(arm.joint1_spec, lp[1])
                arm.set_normalized(n0, n1)
            else:
                parser.error("arm: specify --n0/--n1 or --us0/--us1")
        elif args.cmd == "head":
            head = orch.head
            if args.pose:
                pt = presets_head_pose(args.pose)
                if pt is None:
                    parser.error(f"unknown pose {args.pose!r}")
                head.set_normalized(pt[0], pt[1])
            elif args.pan_us is not None or args.tilt_us is not None:
                if args.pan_us is not None and head.pan_spec is None:
                    parser.error("head: no pan servo wired (--pan-us unsupported)")
                p, t = head.last_pulses
                if args.pan_us is not None:
                    p = args.pan_us
                if args.tilt_us is not None:
                    t = args.tilt_us
                if head.pan_spec is not None and p is None:
                    p = head.pan_spec.center_us
                if t is None:
                    t = head.tilt_spec.center_us
                pan_f = float(p) if p is not None else 0.0
                head.set_pulses(pan_f, float(t))
            elif args.pan is not None or args.tilt is not None:
                if args.pan is not None and head.pan_spec is None:
                    parser.error("head: no pan servo wired (--pan unsupported)")
                lp = head.last_pulses
                pn = (
                    args.pan
                    if args.pan is not None
                    else us_to_normalized(head.pan_spec, lp[0])
                    if head.pan_spec is not None
                    else 0.0
                )
                tn = args.tilt if args.tilt is not None else us_to_normalized(head.tilt_spec, lp[1])
                head.set_normalized(pn, tn)
            else:
                parser.error("head: specify --pose, or --pan/--tilt, or --pan-us/--tilt-us")
        elif args.cmd == "ear":
            ear = orch.ear
            if args.us is not None:
                ear.set_pulse(float(args.us))
            elif args.norm is not None:
                ear.set_normalized(float(args.norm))
            else:
                parser.error("ear: specify -n or --us")
        elif args.cmd == "sequence":
            try:
                result = execute_action(orch, args.name, args.amount)
                print(result)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
        elif args.cmd in ("release", "resume", "reset"):
            # If no joint flags given, apply to all.
            all_joints = not (args.arm or args.head or args.ear)
            if args.cmd == "release":
                if args.arm or all_joints:
                    orch.arm.release()
                if args.head or all_joints:
                    orch.head.release()
                if args.ear or all_joints:
                    orch.ear.release()
            elif args.cmd == "resume":
                if args.arm or all_joints:
                    orch.arm.resume()
                if args.head or all_joints:
                    orch.head.resume()
                if args.ear or all_joints:
                    orch.ear.resume()
            else:
                if args.arm or all_joints:
                    orch.arm.reset()
                if args.head or all_joints:
                    orch.head.reset()
                if args.ear or all_joints:
                    orch.ear.reset()
        return 0
    finally:
        orch.close()


if __name__ == "__main__":
    raise SystemExit(main())
