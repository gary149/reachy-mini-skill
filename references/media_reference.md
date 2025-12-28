# Media Reference

Complete documentation for camera, audio, and media backend configuration.

## Table of Contents
- [Media Backends](#media-backends)
- [Camera System](#camera-system)
- [Audio System](#audio-system)
- [Direction of Arrival](#direction-of-arrival)
- [Supported Hardware](#supported-hardware)

---

## Media Backends

The SDK supports multiple media backends for different use cases.

### Available Backends

| Backend | Value | Description |
|---------|-------|-------------|
| Default | `"default"` | OpenCV + SoundDevice (recommended for most cases) |
| GStreamer | `"gstreamer"` | GStreamer pipeline (Linux, wireless) |
| WebRTC | `"webrtc"` | Remote streaming over WebRTC |
| No Media | `"no_media"` | Disable all media (motion only) |

### Selecting a Backend

```python
# At construction time
robot = ReachyMini(media_backend="default")

# Or let the SDK auto-select based on hardware
robot = ReachyMini()  # Auto-detection
```

### Backend Selection Logic

1. **Local USB connection:** Uses `"default"` (OpenCV + SoundDevice)
2. **Wireless connection:** Uses `"gstreamer"` or `"webrtc"`
3. **Simulation:** Uses simulated camera from MuJoCo
4. **Explicit override:** Always uses specified backend

---

## Camera System

### Accessing the Camera

```python
# Get camera interface
camera = robot.media.camera

# Get current frame
frame = robot.media.get_frame()  # Returns BGR numpy array or None
```

### Camera Properties

```python
# Resolution as (width, height)
width, height = camera.resolution

# Frame rate
fps = camera.framerate

# Intrinsic matrix (3x3)
K = camera.K
# [[fx,  0, cx],
#  [ 0, fy, cy],
#  [ 0,  0,  1]]

# Distortion coefficients (5 values)
D = camera.D
# [k1, k2, p1, p2, k3]

# Full camera specs object
specs = camera.camera_specs
```

### Camera Resolutions

```python
from reachy_mini.media.camera.camera_constants import CameraResolution
```

#### All Available Resolutions

| Enum Value | Resolution | FPS | Notes |
|------------|------------|-----|-------|
| `R1280x720at30fps` | 1280x720 | 30 | Default for Lite |
| `R1280x720at60fps` | 1280x720 | 60 | High frame rate |
| `R1920x1080at30fps` | 1920x1080 | 30 | Full HD |
| `R1920x1080at60fps` | 1920x1080 | 60 | Full HD high FPS |
| `R2304x1296at30fps` | 2304x1296 | 30 | High resolution |
| `R3264x2448at30fps` | 3264x2448 | 30 | Very high resolution |
| `R3840x2592at30fps` | 3840x2592 | 30 | Maximum resolution |
| `R3840x2160at30fps` | 3840x2160 | 30 | 4K UHD |

### Changing Resolution

```python
from reachy_mini.media.camera.camera_constants import CameraResolution

# Set to 1080p
robot.media.camera.set_resolution(CameraResolution.R1920x1080at30fps)

# Set to 720p 60fps for tracking
robot.media.camera.set_resolution(CameraResolution.R1280x720at60fps)
```

### Frame Format

```python
frame = robot.media.get_frame()

# Frame is a numpy array with shape (height, width, 3)
# Color format is BGR (OpenCV standard)
height, width, channels = frame.shape

# Convert to RGB for display
import cv2
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### Undistorting Images

```python
import cv2

frame = robot.media.get_frame()
K = robot.media.camera.K
D = robot.media.camera.D

# Undistort the frame
undistorted = cv2.undistort(frame, K, D)
```

---

## Audio System

### Audio Specifications

| Property | Value |
|----------|-------|
| Sample Rate | 16000 Hz |
| Channels | 2 (stereo) |
| Format | numpy array (float32) |

### Playing Sounds

```python
# Play a built-in sound
robot.media.play_sound("wake_up.wav")

# Available sounds are in the assets/sounds directory
# Common sounds: "wake_up.wav", "sleep.wav"
```

### Recording Audio

```python
# Start recording
robot.media.start_recording()

# Get audio samples (call repeatedly)
sample = robot.media.get_audio_sample()
# Returns numpy array or None

# Stop recording
robot.media.stop_recording()
```

### Streaming Audio Output

```python
import numpy as np

# Start audio output
robot.media.start_playing()

# Push audio samples
audio_data = np.sin(np.linspace(0, 440 * 2 * np.pi, 16000)).astype(np.float32)
robot.media.push_audio_sample(audio_data)

# Stop output
robot.media.stop_playing()
```

### Audio Sample Rates

```python
# Query sample rates
input_rate = robot.media.get_input_audio_samplerate()   # 16000
output_rate = robot.media.get_output_audio_samplerate() # 16000

# Query channels
input_channels = robot.media.get_input_channels()       # 2
output_channels = robot.media.get_output_channels()     # 2
```

---

## Direction of Arrival

The ReSpeaker microphone array provides Direction of Arrival (DoA) detection.

### Getting DoA

```python
result = robot.media.get_DoA()

if result is not None:
    angle, is_valid = result
    # angle: float in radians
    # is_valid: bool indicating measurement quality
```

### DoA Coordinate System

| Angle (radians) | Direction |
|-----------------|-----------|
| 0 | Left side of robot |
| π/2 (~1.57) | Front of robot |
| π (~3.14) | Right side of robot |

### Example: Sound Source Tracking

```python
import math

while True:
    result = robot.media.get_DoA()
    if result and result[1]:  # Valid measurement
        angle, _ = result

        # Convert to yaw for head movement
        # DoA 0 = left, π/2 = front, π = right
        # Head yaw: positive = look left, negative = look right
        yaw_degrees = math.degrees(angle - math.pi/2)

        robot.goto_target(
            head=create_head_pose(yaw=yaw_degrees, degrees=True),
            duration=0.3
        )
```

---

## Supported Hardware

### Cameras

| Hardware | USB ID | Notes |
|----------|--------|-------|
| Lite Camera | 0x38FB:0x1002 | Default camera |
| Arducam | 0x0C45:0x636D | Alternative high-res |
| ReSpeaker Camera | - | Combined audio/video |
| MuJoCo Simulated | - | For simulation mode |

### Microphones

| Hardware | Notes |
|----------|-------|
| ReSpeaker USB Mic Array | Beamforming, DoA support |
| Standard USB Mic | Basic audio input |

### ReSpeaker Requirements

- Firmware version 2.1.0+ for DoA support
- Firmware update utility in `assets/firmware/`

---

## Media Manager Lifecycle

### Initialization

Media is automatically initialized when creating a ReachyMini instance:

```python
with ReachyMini() as robot:
    # Media is ready to use
    frame = robot.media.get_frame()
```

### Manual Cleanup

```python
# Explicitly release media resources
robot.media.close()
```

### Checking Availability

```python
# Check if camera is available
if robot.media.camera is not None:
    frame = robot.media.get_frame()

# Check if audio is available
if robot.media.audio is not None:
    robot.media.play_sound("wake_up.wav")
```

---

## Common Patterns

### Video Recording

```python
import cv2

# Open video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

# Record frames
for _ in range(300):  # 10 seconds at 30fps
    frame = robot.media.get_frame()
    if frame is not None:
        out.write(frame)
    time.sleep(1/30)

out.release()
```

### Face Detection with Look-At

```python
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    frame = robot.media.get_frame()
    if frame is None:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        center_u = x + w // 2
        center_v = y + h // 2
        robot.look_at_image(center_u, center_v, duration=0.2)
```

### Audio Level Monitoring

```python
import numpy as np

robot.media.start_recording()

while True:
    sample = robot.media.get_audio_sample()
    if sample is not None:
        level = np.abs(sample).mean()
        print(f"Audio level: {level:.4f}")

        # React to loud sounds
        if level > 0.5:
            robot.goto_target(antennas=[0.8, 0.8], duration=0.2)
```
