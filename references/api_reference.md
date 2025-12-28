# Reachy Mini API Reference

Complete method signatures and parameter documentation.

## Table of Contents
- [ReachyMini Class](#reachymini-class)
- [Motion Control Methods](#motion-control-methods)
- [State Query Methods](#state-query-methods)
- [Motor Control Methods](#motor-control-methods)
- [Recording Methods](#recording-methods)
- [Media Manager](#media-manager)
- [Utility Functions](#utility-functions)
- [Kinematics Engines](#kinematics-engines)
- [Types and Enums](#types-and-enums)

---

## ReachyMini Class

### Constructor

```python
ReachyMini(
    robot_name: str = "reachy_mini",
    localhost_only: bool = True,
    spawn_daemon: bool = False,
    use_sim: bool = False,
    timeout: float = 5.0,
    automatic_body_yaw: bool = True,
    log_level: str = "INFO",
    media_backend: str = "default"
) -> ReachyMini
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `robot_name` | `str` | `"reachy_mini"` | Identifier for the robot instance |
| `localhost_only` | `bool` | `True` | `True` = connect to localhost daemon only, `False` = use network discovery |
| `spawn_daemon` | `bool` | `False` | Automatically spawn daemon process if not running |
| `use_sim` | `bool` | `False` | Use MuJoCo simulation instead of real robot |
| `timeout` | `float` | `5.0` | Connection timeout in seconds |
| `automatic_body_yaw` | `bool` | `True` | Enable automatic body yaw computation in IK |
| `log_level` | `str` | `"INFO"` | Logging verbosity: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `media_backend` | `str` | `"default"` | Media backend: `"default"`, `"gstreamer"`, `"webrtc"`, `"no_media"` |

### Context Manager

```python
with ReachyMini() as robot:
    # Robot automatically connects and cleans up
    pass
```

---

## Motion Control Methods

### set_target()

Immediately set robot position without interpolation.

```python
set_target(
    head: Optional[np.ndarray] = None,
    antennas: Optional[Union[List[float], np.ndarray]] = None,
    body_yaw: Optional[float] = None
) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `head` | `Optional[np.ndarray]` | 4x4 pose matrix for head position |
| `antennas` | `Optional[List[float]]` | `[right_angle, left_angle]` in radians |
| `body_yaw` | `Optional[float]` | Body yaw angle in radians |

### goto_target()

Smoothly move to target position with interpolation.

```python
goto_target(
    head: Optional[np.ndarray] = None,
    antennas: Optional[Union[List[float], np.ndarray]] = None,
    duration: float = 0.5,
    method: InterpolationTechnique = InterpolationTechnique.MIN_JERK,
    body_yaw: Optional[float] = 0.0
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `head` | `Optional[np.ndarray]` | `None` | 4x4 pose matrix |
| `antennas` | `Optional[List[float]]` | `None` | `[right, left]` radians |
| `duration` | `float` | `0.5` | Motion duration in seconds |
| `method` | `InterpolationTechnique` | `MIN_JERK` | Interpolation method |
| `body_yaw` | `Optional[float]` | `0.0` | Body yaw in radians |

### look_at_image()

Orient head to look at image pixel coordinates.

```python
look_at_image(
    u: int,
    v: int,
    duration: float = 1.0,
    perform_movement: bool = True
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `u` | `int` | - | Horizontal pixel coordinate |
| `v` | `int` | - | Vertical pixel coordinate |
| `duration` | `float` | `1.0` | Movement duration in seconds |
| `perform_movement` | `bool` | `True` | Execute movement or just calculate pose |

**Returns:** 4x4 head pose matrix

### look_at_world()

Orient head to look at 3D world coordinates.

```python
look_at_world(
    x: float,
    y: float,
    z: float,
    duration: float = 1.0,
    perform_movement: bool = True
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `float` | - | X coordinate in meters |
| `y` | `float` | - | Y coordinate in meters |
| `z` | `float` | - | Z coordinate in meters |
| `duration` | `float` | `1.0` | Movement duration in seconds |
| `perform_movement` | `bool` | `True` | Execute movement or just calculate pose |

**Returns:** 4x4 head pose matrix

### wake_up()

Execute wake-up behavior with animation and sound.

```python
wake_up() -> None
```

### goto_sleep()

Execute sleep behavior with animation and sound.

```python
goto_sleep() -> None
```

### play_move()

Play a recorded motion.

```python
play_move(
    move: Move,
    play_frequency: float = 100.0,
    initial_goto_duration: float = 0.0,
    sound: bool = True
) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `move` | `Move` | - | Move object to play |
| `play_frequency` | `float` | `100.0` | Evaluation frequency in Hz |
| `initial_goto_duration` | `float` | `0.0` | Duration to reach starting position |
| `sound` | `bool` | `True` | Play associated sound if available |

### async_play_move()

Async version of play_move.

```python
async async_play_move(
    move: Move,
    play_frequency: float = 100.0,
    initial_goto_duration: float = 0.0,
    sound: bool = True
) -> None
```

---

## State Query Methods

### get_current_head_pose()

Get current head pose as 4x4 transformation matrix.

```python
get_current_head_pose() -> np.ndarray
```

**Returns:** 4x4 numpy array (transformation matrix)

### get_current_joint_positions()

Get all current joint positions.

```python
get_current_joint_positions() -> Tuple[List[float], List[float]]
```

**Returns:** Tuple of `(head_joints, antenna_joints)`
- `head_joints`: List of 7 floats (body_rotation + 6 stewart platform motors)
- `antenna_joints`: List of 2 floats `[right, left]`

### get_present_antenna_joint_positions()

Get current antenna positions only.

```python
get_present_antenna_joint_positions() -> List[float]
```

**Returns:** List of 2 floats `[right_antenna, left_antenna]` in radians

---

## Motor Control Methods

### enable_motors()

Enable motor torque.

```python
enable_motors(ids: Optional[List[str]] = None) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ids` | `Optional[List[str]]` | `None` | Motor IDs to enable, or `None` for all |

### disable_motors()

Disable motor torque.

```python
disable_motors(ids: Optional[List[str]] = None) -> None
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ids` | `Optional[List[str]]` | `None` | Motor IDs to disable, or `None` for all |

**Motor IDs:**
- `"body_rotation"` - Body yaw motor
- `"stewart_1"` through `"stewart_6"` - Stewart platform motors
- `"right_antenna"` - Right antenna motor
- `"left_antenna"` - Left antenna motor

### enable_gravity_compensation()

Enable gravity compensation mode (requires Placo kinematics).

```python
enable_gravity_compensation() -> None
```

### disable_gravity_compensation()

Disable gravity compensation mode.

```python
disable_gravity_compensation() -> None
```

---

## Recording Methods

### start_recording()

Start recording robot motion.

```python
start_recording() -> None
```

### stop_recording()

Stop recording and return recorded data.

```python
stop_recording() -> Optional[List[Dict[str, float | List[float] | List[List[float]]]]]
```

**Returns:** List of recorded frames, each containing:
- `timestamp`: Time in seconds
- `head_pose`: 4x4 matrix as nested list
- `joint_positions`: Joint angles
- Other motion data

---

## Media Manager

Access via `robot.media`.

### Properties

```python
media.camera: Optional[CameraBase]  # Camera interface
media.audio: Optional[AudioBase]    # Audio interface
media.backend: MediaBackend         # Current backend type
```

### get_frame()

Get current camera frame.

```python
media.get_frame() -> Optional[np.ndarray]
```

**Returns:** BGR numpy array with shape `(height, width, 3)` or `None`

### Camera Properties

```python
media.camera.resolution -> Tuple[int, int]  # (width, height)
media.camera.framerate -> int               # FPS
media.camera.K -> Optional[np.ndarray]      # 3x3 intrinsic matrix
media.camera.D -> Optional[np.ndarray]      # Distortion coefficients (5,)
media.camera.set_resolution(resolution: CameraResolution) -> None
```

### Audio Methods

```python
media.play_sound(sound_file: str) -> None
media.start_recording() -> None
media.get_audio_sample() -> Optional[np.ndarray]
media.stop_recording() -> None

media.start_playing() -> None
media.push_audio_sample(data: np.ndarray) -> None
media.stop_playing() -> None

media.get_input_audio_samplerate() -> int   # Default: 16000 Hz
media.get_output_audio_samplerate() -> int  # Default: 16000 Hz
media.get_input_channels() -> int           # Default: 2
media.get_output_channels() -> int          # Default: 2

media.get_DoA() -> Optional[Tuple[float, bool]]  # (angle_radians, is_valid)
```

### close()

Release all media resources.

```python
media.close() -> None
```

---

## Utility Functions

### create_head_pose()

Create a 4x4 pose matrix from position and rotation.

```python
from reachy_mini.utils import create_head_pose

create_head_pose(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    roll: float = 0,
    pitch: float = 0,
    yaw: float = 0,
    mm: bool = False,
    degrees: bool = True
) -> np.ndarray
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `float` | `0` | X position (meters unless `mm=True`) |
| `y` | `float` | `0` | Y position |
| `z` | `float` | `0` | Z position |
| `roll` | `float` | `0` | Roll angle |
| `pitch` | `float` | `0` | Pitch angle |
| `yaw` | `float` | `0` | Yaw angle |
| `mm` | `bool` | `False` | Interpret x/y/z as millimeters |
| `degrees` | `bool` | `True` | Interpret angles as degrees |

**Returns:** 4x4 numpy transformation matrix

---

## Kinematics Engines

All kinematics engines implement:

```python
ik(pose: np.ndarray, body_yaw: float = 0.0, check_collision: bool = False) -> np.ndarray
fk(joint_angles: np.ndarray) -> np.ndarray
```

### AnalyticalKinematics

```python
from reachy_mini.kinematics.analytical import AnalyticalKinematics
kin = AnalyticalKinematics()
```

Always available. Uses Rust bindings for fast analytical IK.

### PlacoKinematics

```python
from reachy_mini.kinematics.placo import PlacoKinematics
kin = PlacoKinematics()
```

Requires: `pip install reachy_mini[placo_kinematics]`
Features: Collision checking, gravity compensation

### NNKinematics

```python
from reachy_mini.kinematics.nn import NNKinematics
kin = NNKinematics()
```

Requires: `pip install reachy_mini[nn_kinematics]`
Features: Neural network based, very fast

---

## Types and Enums

### InterpolationTechnique

```python
from reachy_mini.motion.goto_move import InterpolationTechnique

InterpolationTechnique.LINEAR       # Linear interpolation
InterpolationTechnique.MIN_JERK     # Minimum jerk (default, smoothest)
InterpolationTechnique.EASE_IN_OUT  # Ease in/out
InterpolationTechnique.CARTOON      # Cartoon-style animation
```

### CameraResolution

```python
from reachy_mini.media.camera.camera_constants import CameraResolution

CameraResolution.R1280x720at30fps   # Default for Lite
CameraResolution.R1280x720at60fps
CameraResolution.R1920x1080at30fps
CameraResolution.R1920x1080at60fps
CameraResolution.R2304x1296at30fps
CameraResolution.R3264x2448at30fps
CameraResolution.R3840x2592at30fps
CameraResolution.R3840x2160at30fps
```

### MotorControlMode

```python
from reachy_mini.io.models import MotorControlMode

MotorControlMode.Enabled               # Normal position control
MotorControlMode.Disabled              # Torque OFF
MotorControlMode.GravityCompensation   # Gravity compensation mode
```

### XYZRPYPose

```python
from reachy_mini.io.models import XYZRPYPose

pose = XYZRPYPose(
    x=0.0,      # meters
    y=0.0,      # meters
    z=0.0,      # meters
    roll=0.0,   # radians
    pitch=0.0,  # radians
    yaw=0.0     # radians
)
```

---

## Constants

```python
from reachy_mini.utils import URDF_ROOT_PATH, ASSETS_ROOT_PATH, MODELS_ROOT_PATH

URDF_ROOT_PATH   # Path to robot URDF files
ASSETS_ROOT_PATH # Path to assets (sounds, etc.)
MODELS_ROOT_PATH # Path to kinematics models
```

---

## Common Exceptions

| Exception | Description |
|-----------|-------------|
| `TimeoutError` | Connection or task timeout |
| `ConnectionError` | Lost connection to daemon |
| `ValueError` | Invalid parameters |
| `RuntimeError` | Camera/audio not initialized |
| `ImportError` | Optional kinematics engine not installed |
