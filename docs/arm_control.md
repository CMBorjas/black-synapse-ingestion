# Arm Control — How It Works

## Overview

The robot's arm is controlled through a chain that starts with a spoken or typed instruction and ends with physical servo motion. The LLM sits in the middle — it understands intent and picks an action, but never touches raw hardware values.

```
User speech / text
      │
      ▼
   ASR (Whisper)
      │  transcript
      ▼
   n8n LangChain agent  ←──── Qdrant knowledge base (RAG tools)
      │  decides to move arm
      ▼
   control_arm tool  (toolHttpRequest)
      │  POST {"action": "wave"}
      ▼
   Worker  /arm/action  (FastAPI, port 8000)
      │  dispatches to arm_actions.py
      ▼
   Pose / Sequence  (action_servos/sequences.py)
      │  calculates µs pulse values
      ▼
   ArmController  (action_servos/groups.py)
      │  writes to PCA9685 over I2C
      ▼
   Servo motors
```

---

## Layer 1 — The LLM (n8n + Ollama)

The LLM is Qwen 2.5 7B running locally via Ollama. It is orchestrated by an n8n LangChain agent that gives it a set of registered tools. One of those tools is `control_arm`.

The tool description is plain English — the LLM reads it and decides when and how to use it:

```
Control the robot's physical arm. Use this when the user asks you to move, wave,
point, extend, retract, or rest the arm, or when a gesture would enhance the interaction.

Actions:
- extend: unfold arm outward (requires amount: 0–100, where 100 = fully extended)
- retract: fold arm inward (requires amount: 0–100, where 100 = fully retracted)
- wave: perform a friendly wave (no amount needed)
- point: point forward (no amount needed)
- rest: move arm to safe resting position (no amount needed)
- release: release motor torque so arm can be moved by hand (no amount needed)
```

The LLM never sees pulse widths, channel numbers, or normalized values. It only knows about these six semantic actions.

When the LLM decides to use the tool, n8n sends an HTTP request:

```
POST http://host.docker.internal:8000/arm/action
Content-Type: application/json

{"action": "wave"}
```

or, for a parameterized action:

```json
{"action": "extend", "amount": 75}
```

The worker returns a plain English result that the LLM reads back into its response:

```json
{"success": true, "action": "wave", "result": "Waved."}
```

---

## Layer 2 — The Worker Endpoint (`worker/app/main.py`)

The FastAPI worker (port 8000) exposes `POST /arm/action`. On startup it opens a `ServoOrchestrator` singleton — a single persistent connection to the PCA9685 servo driver over I2C. This stays open for the lifetime of the worker process.

```python
_arm_orch = ServoOrchestrator()
_arm_orch.open(bus=7, address=0x40, frequency_hz=50.0)
```

If the hardware is not available (e.g. running on a dev machine without the robot), `_arm_orch` is set to `None` and the endpoint returns `503` instead of crashing the worker.

The endpoint validates the request with Pydantic:

```python
class ArmActionRequest(BaseModel):
    action: str           # "extend" | "retract" | "wave" | "point" | "rest" | "release"
    amount: Optional[int] # 0–100, only for extend/retract
```

Because servo ramps block (they sleep between steps), the action runs in a thread executor so it does not block the async event loop:

```python
description = await loop.run_in_executor(
    None, execute_action, _arm_orch, request.action, request.amount
)
```

---

## Layer 3 — The Action Library (`worker/app/arm_actions.py`)

This is where semantic actions are translated into actual motion. It is the only place in the codebase where arm geometry is encoded.

At the top of the file, calibration constants define key positions as normalized values (−1 to +1, where −1 = minimum pulse, 0 = center, +1 = maximum pulse):

```python
_EXTEND_J0  =  0.8   # shoulder fully extended
_EXTEND_J1  =  0.7   # elbow fully extended
_RETRACT_J0 = -0.5   # shoulder fully retracted
_RETRACT_J1 = -0.6   # elbow fully retracted
_REST_J0    = -0.3   # shoulder at rest
_REST_J1    = -0.5   # elbow at rest
_POINT_J0   =  0.6   # shoulder for pointing
_POINT_J1   =  0.2   # elbow for pointing
```

**Parameterized actions** (`extend`, `retract`) interpolate linearly from neutral (0%) to the calibrated endpoint (100%):

```python
def _extend_pose(orch, amount):
    t = amount / 100.0          # 0.0 → 1.0
    j0 = lerp(0.0, _EXTEND_J0, t)
    j1 = lerp(0.0, _EXTEND_J1, t)
    return Pose(arm_j0=..., arm_j1=...)
```

**Named gestures** (`wave`) are multi-keyframe `Sequence` objects:

```python
Sequence()
  .add(p(0.5, 0.3), duration_s=0.40)  # raise up
  .add(p(0.5, 0.7), duration_s=0.25)  # wave out
  .add(p(0.5, 0.1), duration_s=0.25)  # wave in
  .add(p(0.5, 0.7), duration_s=0.25)  # wave out
  .add(p(0.5, 0.1), duration_s=0.25)  # wave in
  .add(p(0.0, 0.0), duration_s=0.40)  # return to neutral
```

The `execute_action()` function dispatches by name and returns a human-readable string:

| Call | Returns |
|------|---------|
| `execute_action(orch, "extend", 75)` | `"Extended arm to 75%."` |
| `execute_action(orch, "wave", None)` | `"Waved."` |
| `execute_action(orch, "spin", None)` | raises `ValueError` → HTTP 400 |

---

## Layer 4 — Sequences and Poses (`action_servos/sequences.py`)

A `Pose` describes a target state for any subset of joints, in microseconds:

```python
Pose(arm_j0=1740.0, arm_j1=1710.0)  # shoulder + elbow target
```

A `Sequence` is an ordered list of `Keyframe` objects, each pairing a `Pose` with a ramp duration. Calling `.play(orch)` executes them in order:

```python
for keyframe in sequence.keyframes:
    _ramp_to_pose(orch, keyframe.pose, keyframe.duration_s, keyframe.steps)
```

Each ramp interpolates **all moving joints simultaneously** in a single timed loop, so the arm moves smoothly in one motion rather than one joint at a time:

```python
for i in range(1, steps + 1):
    a = i / steps
    arm.set_pulses(start_j0 + (target_j0 - start_j0) * a,
                   start_j1 + (target_j1 - start_j1) * a)
    time.sleep(duration_s / steps)
```

---

## Layer 5 — The Hardware (`action_servos/hardware.py` + `groups.py`)

`ArmController` holds a reference to a `PCA9685` object and tracks the last known pulse for each joint. When `set_pulses()` is called it converts microseconds to a 12-bit PWM count and writes it to the chip over I2C:

```
pulse_us  →  off_count = round(pulse_us × 4096 × 50Hz / 1,000,000)
off_count  →  4-byte I2C block write to LEDn_ON / LEDn_OFF registers
```

The PCA9685 runs at 50 Hz (standard hobby servo frequency). Each of its 16 channels independently controls one servo. The arm uses:

| Joint | Channel | Range |
|-------|---------|-------|
| Shoulder (j0) | 6 | 900–2100 µs |
| Elbow (j1) | 5 | 900–2100 µs |

The chip is on I2C bus 7 (`/dev/i2c-7`) at address `0x40`. Bus 1 is not used because it is permanently claimed by the Jetson's onboard INA3221 power monitor at the same address.

---

## Calibration

To adjust how the arm moves, edit the constants at the top of `worker/app/arm_actions.py`. No other file needs to change.

1. Run the worker and `curl` an action to observe the motion
2. Adjust the relevant constant (`_EXTEND_J0`, `_WAVE_*`, etc.)
3. Restart the worker and re-test

For raw manual testing without the LLM:

```bash
# Move arm directly
python -m action_servos arm --n0 0.5 --n1 -0.3

# Test a named action via HTTP
curl -X POST http://localhost:8000/arm/action \
  -H "Content-Type: application/json" \
  -d '{"action": "extend", "amount": 50}'
```

---

## Adding a New Action

1. Define the target pose(s) as calibration constants in `arm_actions.py`
2. Add a new branch in `execute_action()` with the action name and motion logic
3. Update the `control_arm` tool description in n8n so the LLM knows the action exists

No changes to the hardware layer, sequences layer, or worker endpoint are needed.
