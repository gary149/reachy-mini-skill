# Motion Reference

Detailed documentation for motion control, interpolation, and move playback.

## Table of Contents
- [Interpolation Techniques](#interpolation-techniques)
- [Move Classes](#move-classes)
- [GotoMove](#gotomove)
- [RecordedMove](#recordedmove)
- [RecordedMoves Library](#recordedmoves-library)
- [Interpolation Functions](#interpolation-functions)
- [Motion Recording Format](#motion-recording-format)

---

## Interpolation Techniques

```python
from reachy_mini.motion.goto_move import InterpolationTechnique
```

### LINEAR

Simple linear interpolation between start and end positions.
- **Use case:** Fast, predictable movements
- **Characteristics:** Constant velocity, abrupt start/stop

### MIN_JERK (Default)

Minimum jerk trajectory for smooth, natural motion.
- **Use case:** Most movements, especially expressive ones
- **Characteristics:** Smooth acceleration/deceleration, natural appearance
- **Formula:** 5th order polynomial minimizing jerk (rate of change of acceleration)

### EASE_IN_OUT

Smooth easing at both ends of the motion.
- **Use case:** Gentle transitions
- **Characteristics:** Slow start, faster middle, slow end

### CARTOON

Exaggerated cartoon-style animation.
- **Use case:** Playful, expressive movements
- **Characteristics:** Anticipation, overshoot, follow-through

---

## Move Classes

All move classes inherit from the abstract `Move` base class.

### Move Base Class

```python
from abc import ABC, abstractmethod

class Move(ABC):
    @property
    @abstractmethod
    def duration(self) -> float:
        """Total duration in seconds."""
        pass

    @property
    def sound_path(self) -> Optional[Path]:
        """Optional path to sound file played with move."""
        return None

    @abstractmethod
    def evaluate(self, t: float) -> Tuple[
        Optional[np.ndarray],  # Head pose (4x4 matrix)
        Optional[np.ndarray],  # Antennas [right, left] in radians
        Optional[float]        # Body yaw in radians
    ]:
        """Evaluate move at time t (0 to duration)."""
        pass
```

---

## GotoMove

Interpolated movement between two poses.

### Constructor

```python
from reachy_mini.motion.goto_move import GotoMove

move = GotoMove(
    start_head_pose: np.ndarray,           # Starting 4x4 pose
    target_head_pose: Optional[np.ndarray], # Target 4x4 pose (None = no change)
    start_antennas: np.ndarray,            # Starting [right, left]
    target_antennas: Optional[np.ndarray], # Target [right, left] (None = no change)
    start_body_yaw: float,                 # Starting body yaw
    target_body_yaw: Optional[float],      # Target body yaw (None = no change)
    duration: float,                       # Duration in seconds
    method: InterpolationTechnique         # Interpolation method
)
```

### Example

```python
import numpy as np
from reachy_mini.motion.goto_move import GotoMove, InterpolationTechnique
from reachy_mini.utils import create_head_pose

start_pose = create_head_pose(yaw=0, degrees=True)
end_pose = create_head_pose(yaw=30, degrees=True)

move = GotoMove(
    start_head_pose=start_pose,
    target_head_pose=end_pose,
    start_antennas=np.array([0.0, 0.0]),
    target_antennas=np.array([0.5, 0.5]),
    start_body_yaw=0.0,
    target_body_yaw=None,  # Don't change body yaw
    duration=1.0,
    method=InterpolationTechnique.MIN_JERK
)

# Evaluate at t=0.5 seconds
head, antennas, body_yaw = move.evaluate(0.5)
```

---

## RecordedMove

Playback of a previously recorded motion.

### Constructor

```python
from reachy_mini.motion.recorded_move import RecordedMove

move = RecordedMove(
    move: Dict[str, Any],           # Recorded motion data
    sound_path: Optional[Path] = None  # Optional sound file
)
```

### Move Data Format

```python
{
    "duration": 5.0,  # Total duration in seconds
    "frames": [
        {
            "timestamp": 0.0,
            "head_pose": [[...], [...], [...], [...]],  # 4x4 matrix
            "antennas": [0.1, 0.1],  # [right, left]
            "body_yaw": 0.0
        },
        # ... more frames
    ]
}
```

---

## RecordedMoves Library

Load and play moves from HuggingFace Hub datasets.

### Constructor

```python
from reachy_mini.motion.recorded_move import RecordedMoves

moves = RecordedMoves(hf_dataset_name: str)
```

### Methods

```python
# List all available move names
move_names = moves.list_moves()  # Returns List[str]

# Get a specific move
move = moves.get(move_name: str)  # Returns RecordedMove
```

### Example

```python
from reachy_mini.motion.recorded_move import RecordedMoves

# Load the official dance library
moves = RecordedMoves("pollen-robotics/reachy-mini-dances-library")

# See what's available
print(moves.list_moves())
# ['wave', 'dance1', 'nod', 'shake_head', ...]

# Get and play a move
wave = moves.get("wave")
robot.play_move(wave, initial_goto_duration=0.5)
```

### Known Move Libraries

| Dataset | Description |
|---------|-------------|
| `pollen-robotics/reachy-mini-dances-library` | Official dance and gesture library |

---

## Interpolation Functions

Low-level interpolation utilities.

### minimum_jerk()

Create a minimum jerk trajectory function.

```python
from reachy_mini.utils.interpolation import minimum_jerk

trajectory = minimum_jerk(
    starting_position: np.ndarray,          # Start position
    goal_position: np.ndarray,              # End position
    duration: float,                        # Duration in seconds
    starting_velocity: Optional[np.ndarray] = None,
    starting_acceleration: Optional[np.ndarray] = None,
    final_velocity: Optional[np.ndarray] = None,
    final_acceleration: Optional[np.ndarray] = None
) -> Callable[[float], np.ndarray]

# Usage
position_at_t = trajectory(t)  # t in [0, duration]
```

### linear_pose_interpolation()

Interpolate between two 4x4 pose matrices.

```python
from reachy_mini.utils.interpolation import linear_pose_interpolation

interpolated_pose = linear_pose_interpolation(
    start_pose: np.ndarray,   # 4x4 start pose
    target_pose: np.ndarray,  # 4x4 target pose
    t: float                  # Interpolation factor (0 to 1)
) -> np.ndarray  # Returns 4x4 interpolated pose
```

### time_trajectory()

Apply time warping for different interpolation feels.

```python
from reachy_mini.utils.interpolation import time_trajectory

warped_t = time_trajectory(
    t: float,                                    # Normalized time (0 to 1)
    method: InterpolationTechnique = InterpolationTechnique.MIN_JERK
) -> float  # Returns warped time value (0 to 1)
```

---

## Motion Recording Format

When using `robot.start_recording()` / `robot.stop_recording()`:

### Recorded Data Structure

```python
recorded_data = robot.stop_recording()
# Returns: List[Dict]

# Each frame contains:
{
    "timestamp": float,           # Time in seconds from start
    "head_pose": List[List[float]],  # 4x4 matrix as nested list
    "head_joints": List[float],   # 7 joint angles
    "antennas": List[float],      # [right, left] in radians
    "body_yaw": float             # Body yaw in radians
}
```

### Saving Recordings

```python
import json

robot.start_recording()
# ... perform movements ...
data = robot.stop_recording()

# Save to file
with open("my_move.json", "w") as f:
    json.dump({"frames": data, "duration": data[-1]["timestamp"]}, f)

# Load later
with open("my_move.json", "r") as f:
    move_data = json.load(f)

move = RecordedMove(move_data)
robot.play_move(move)
```

---

## Playback Parameters

### play_frequency

Controls how often the move is evaluated during playback.

```python
robot.play_move(move, play_frequency=100.0)  # 100 Hz (default)
robot.play_move(move, play_frequency=50.0)   # 50 Hz (smoother for slow moves)
```

- Higher = smoother motion, more CPU usage
- Default 100 Hz is suitable for most moves
- Reduce for slow, gentle movements

### initial_goto_duration

Time to smoothly transition from current position to move's start position.

```python
robot.play_move(move, initial_goto_duration=0.0)  # Jump to start (default)
robot.play_move(move, initial_goto_duration=1.0)  # 1 second transition
```

- Use `0.0` when already in position
- Use `0.5-2.0` for smooth transitions from arbitrary positions

### sound

Whether to play the move's associated sound file.

```python
robot.play_move(move, sound=True)   # Play sound (default)
robot.play_move(move, sound=False)  # Silent playback
```

---

## Custom Move Creation

Create custom moves by subclassing `Move`:

```python
from reachy_mini.motion.move import Move
import numpy as np
from typing import Optional, Tuple
from pathlib import Path

class CircleMove(Move):
    def __init__(self, radius: float, duration: float):
        self._radius = radius
        self._duration = duration

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def sound_path(self) -> Optional[Path]:
        return None

    def evaluate(self, t: float) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[float]
    ]:
        angle = (t / self._duration) * 2 * np.pi
        yaw = self._radius * np.sin(angle)
        pitch = self._radius * np.cos(angle)

        from reachy_mini.utils import create_head_pose
        pose = create_head_pose(yaw=np.degrees(yaw), pitch=np.degrees(pitch))

        return pose, None, None  # Only move head, not antennas/body

# Usage
circle = CircleMove(radius=0.3, duration=3.0)
robot.play_move(circle)
```
