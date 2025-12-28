# Application Patterns

Advanced patterns for building production-grade applications with Reachy Mini, derived from the conversation app.

## Table of Contents
- [Movement Manager](#movement-manager)
- [Layered Motion System](#layered-motion-system)
- [Audio-Reactive Motion](#audio-reactive-motion)
- [Face Tracking](#face-tracking)
- [Tool System](#tool-system)
- [OpenAI Realtime Integration](#openai-realtime-integration)
- [Profile System](#profile-system)
- [Worker Thread Patterns](#worker-thread-patterns)

---

## Movement Manager

A centralized motion controller running at 100 Hz for smooth, coordinated movement.

### Architecture

```python
import threading
import time
from queue import Queue
from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np

@dataclass
class MovementState:
    current_head_pose: np.ndarray
    current_antennas: np.ndarray
    current_body_yaw: float
    is_moving: bool = False
    is_listening: bool = False

class MovementManager:
    def __init__(self, robot: ReachyMini):
        self.robot = robot
        self.state = MovementState(
            current_head_pose=np.eye(4),
            current_antennas=np.array([0.0, 0.0]),
            current_body_yaw=0.0
        )
        self._command_queue: Queue = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Secondary offsets (additive)
        self._speech_offset = np.zeros(6)  # x, y, z, roll, pitch, yaw
        self._tracking_offset = np.zeros(6)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _control_loop(self):
        """100 Hz control loop."""
        period = 1.0 / 100.0
        next_time = time.monotonic()

        while self._running:
            # Process commands
            while not self._command_queue.empty():
                cmd = self._command_queue.get_nowait()
                self._process_command(cmd)

            # Compose final pose from primary + secondary
            final_pose = self._compose_pose()

            # Send to robot
            self.robot.set_target(
                head=final_pose,
                antennas=self.state.current_antennas,
                body_yaw=self.state.current_body_yaw
            )

            # Phase-aligned timing
            next_time += period
            sleep_time = next_time - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _compose_pose(self) -> np.ndarray:
        """Combine primary pose with secondary offsets."""
        pose = self.state.current_head_pose.copy()

        with self._lock:
            # Add speech and tracking offsets
            total_offset = self._speech_offset + self._tracking_offset

        # Apply offset as world-frame translation + rotation
        from reachy_mini.utils import create_head_pose
        offset_pose = create_head_pose(
            x=total_offset[0] / 1000,  # mm to m
            y=total_offset[1] / 1000,
            z=total_offset[2] / 1000,
            roll=np.degrees(total_offset[3]),
            pitch=np.degrees(total_offset[4]),
            yaw=np.degrees(total_offset[5]),
            degrees=True
        )

        return offset_pose @ pose

    def set_speech_offsets(self, x_mm=0, y_mm=0, z_mm=0, roll=0, pitch=0, yaw=0):
        """Set speech-reactive offsets (called from audio thread)."""
        with self._lock:
            self._speech_offset = np.array([x_mm, y_mm, z_mm, roll, pitch, yaw])

    def set_tracking_offsets(self, x_mm=0, y_mm=0, z_mm=0, roll=0, pitch=0, yaw=0):
        """Set face-tracking offsets (called from vision thread)."""
        with self._lock:
            self._tracking_offset = np.array([x_mm, y_mm, z_mm, roll, pitch, yaw])

    def queue_move(self, move):
        """Queue a Move object for execution."""
        self._command_queue.put(("play_move", move))

    def set_listening(self, listening: bool):
        """Set listening state (freezes antennas)."""
        self._command_queue.put(("set_listening", listening))
```

### Usage

```python
with ReachyMini() as robot:
    manager = MovementManager(robot)
    manager.start()

    # Queue moves
    manager.queue_move(some_dance)

    # Set real-time offsets from other threads
    manager.set_speech_offsets(yaw=0.05)  # Speech wobble
    manager.set_tracking_offsets(pitch=-0.1)  # Look at face

    manager.stop()
```

---

## Layered Motion System

Separate primary (exclusive) and secondary (additive) motion layers.

### Primary Moves

Mutually exclusive - only one runs at a time:
- **Emotions**: Pre-recorded expressions
- **Dances**: Choreographed sequences
- **Goto**: Smooth transitions to poses
- **Breathing**: Idle animation

### Secondary Offsets

Additive layers that blend with primary:
- **Speech wobble**: Audio-reactive head movement
- **Face tracking**: Follow detected faces

### Breathing Move (Idle Animation)

```python
import math
import time
from typing import Optional, Tuple
import numpy as np

class BreathingMove:
    """Gentle idle animation when robot is inactive."""

    def __init__(
        self,
        start_pose: np.ndarray,
        start_antennas: np.ndarray,
        breath_amplitude_mm: float = 5.0,
        breath_frequency_hz: float = 0.2,
        antenna_amplitude_deg: float = 15.0,
        antenna_frequency_hz: float = 0.5,
        interpolation_duration: float = 1.0
    ):
        self.start_pose = start_pose
        self.start_antennas = start_antennas
        self.breath_amp = breath_amplitude_mm
        self.breath_freq = breath_frequency_hz
        self.antenna_amp = math.radians(antenna_amplitude_deg)
        self.antenna_freq = antenna_frequency_hz
        self.interp_duration = interpolation_duration

        # Neutral pose to interpolate toward
        self.neutral_pose = np.eye(4)
        self.neutral_antennas = np.array([0.3, 0.3])  # Slightly open

        self.start_time = time.monotonic()

    @property
    def duration(self) -> float:
        return float('inf')  # Runs until interrupted

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        # Interpolation factor (0 to 1 over interp_duration)
        alpha = min(1.0, t / self.interp_duration)

        # Interpolate toward neutral
        pose = self._lerp_pose(self.start_pose, self.neutral_pose, alpha)
        antennas = self.start_antennas + alpha * (self.neutral_antennas - self.start_antennas)

        # Add breathing oscillation (Z-axis)
        breath_z = self.breath_amp * math.sin(2 * math.pi * self.breath_freq * t)
        pose[2, 3] += breath_z / 1000  # mm to m

        # Add antenna sway
        antenna_sway = self.antenna_amp * math.sin(2 * math.pi * self.antenna_freq * t)
        antennas = antennas + np.array([antenna_sway, -antenna_sway])

        return pose, antennas, None

    def _lerp_pose(self, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        """Linear interpolation between poses."""
        from scipy.spatial.transform import Rotation, Slerp

        # Interpolate translation
        trans = p1[:3, 3] + t * (p2[:3, 3] - p1[:3, 3])

        # Slerp rotation
        r1 = Rotation.from_matrix(p1[:3, :3])
        r2 = Rotation.from_matrix(p2[:3, :3])
        slerp = Slerp([0, 1], Rotation.concatenate([r1, r2]))
        rot = slerp(t).as_matrix()

        result = np.eye(4)
        result[:3, :3] = rot
        result[:3, 3] = trans
        return result
```

### Move Queue Pattern

```python
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class QueuedMove:
    move: "Move"
    start_time: Optional[float] = None

class MoveQueue:
    def __init__(self):
        self._queue: deque[QueuedMove] = deque()
        self._current: Optional[QueuedMove] = None

    def enqueue(self, move: "Move"):
        self._queue.append(QueuedMove(move=move))

    def clear(self):
        self._queue.clear()
        self._current = None

    def get_current_output(self, now: float):
        """Get pose from current move, advancing queue as needed."""
        # Start next move if none active
        if self._current is None and self._queue:
            self._current = self._queue.popleft()
            self._current.start_time = now

        if self._current is None:
            return None, None, None

        # Evaluate current move
        t = now - self._current.start_time
        if t > self._current.move.duration:
            # Move finished, advance queue
            self._current = None
            return self.get_current_output(now)

        return self._current.move.evaluate(t)
```

---

## Audio-Reactive Motion

Convert audio to head movement for lifelike speech animation.

### Head Wobbler

```python
import base64
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass
class AudioChunk:
    samples: np.ndarray
    play_time: float

class HeadWobbler:
    """Convert audio deltas to head movement offsets."""

    SAMPLE_RATE = 24000  # OpenAI realtime uses 24kHz
    LATENCY_COMPENSATION = 0.08  # seconds

    # Movement parameters
    YAW_AMPLITUDE = 0.03  # radians
    PITCH_AMPLITUDE = 0.02
    ROLL_AMPLITUDE = 0.01

    def __init__(self):
        self._audio_queue: deque[AudioChunk] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._generation = 0  # For session resets

    def push_audio_delta(self, base64_audio: str, generation: int):
        """Push audio delta from OpenAI realtime API."""
        if generation != self._generation:
            return  # Stale audio from old session

        # Decode base64 PCM16
        raw = base64.b64decode(base64_audio)
        samples = np.array(struct.unpack(f'<{len(raw)//2}h', raw), dtype=np.float32)
        samples /= 32768.0  # Normalize to [-1, 1]

        # Schedule playback with latency compensation
        duration = len(samples) / self.SAMPLE_RATE
        play_time = time.monotonic() + self.LATENCY_COMPENSATION

        with self._lock:
            self._audio_queue.append(AudioChunk(samples, play_time))

    def get_offsets(self) -> tuple:
        """Get current movement offsets based on audio."""
        now = time.monotonic()

        with self._lock:
            # Remove old chunks
            while self._audio_queue and self._audio_queue[0].play_time + 0.1 < now:
                self._audio_queue.popleft()

            if not self._audio_queue:
                return 0, 0, 0, 0, 0, 0

            # Find current chunk
            current_chunk = None
            for chunk in self._audio_queue:
                chunk_end = chunk.play_time + len(chunk.samples) / self.SAMPLE_RATE
                if chunk.play_time <= now < chunk_end:
                    current_chunk = chunk
                    break

        if current_chunk is None:
            return 0, 0, 0, 0, 0, 0

        # Get amplitude at current time
        offset_samples = int((now - current_chunk.play_time) * self.SAMPLE_RATE)
        window_size = min(512, len(current_chunk.samples) - offset_samples)

        if window_size <= 0:
            return 0, 0, 0, 0, 0, 0

        window = current_chunk.samples[offset_samples:offset_samples + window_size]
        amplitude = np.sqrt(np.mean(window ** 2))  # RMS

        # Convert to movement (simplified - real impl uses frequency analysis)
        t = now * 2 * np.pi
        yaw = self.YAW_AMPLITUDE * amplitude * np.sin(t * 3.5)
        pitch = self.PITCH_AMPLITUDE * amplitude * np.sin(t * 2.1)
        roll = self.ROLL_AMPLITUDE * amplitude * np.sin(t * 4.2)

        return 0, 0, 0, roll, pitch, yaw

    def reset(self):
        """Reset for new session."""
        with self._lock:
            self._audio_queue.clear()
            self._generation += 1
        return self._generation
```

### Integration with Movement Manager

```python
# In audio callback (e.g., OpenAI realtime handler)
def on_audio_delta(delta_base64: str):
    wobbler.push_audio_delta(delta_base64, current_generation)

# In control loop
def control_loop():
    while running:
        offsets = wobbler.get_offsets()
        movement_manager.set_speech_offsets(*offsets)
        time.sleep(0.01)
```

---

## Face Tracking

Track faces and generate head movement offsets.

### YOLO Head Tracker

```python
from typing import Optional, Tuple
import numpy as np

class YOLOHeadTracker:
    """Face tracking using YOLOv8 from Ultralytics."""

    def __init__(self, model_path: str = "arnabdhar/YOLOv8-Face-Detection"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence_threshold = 0.5

    def detect(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect face and return normalized coordinates.

        Returns:
            (x, y) in range [-1, 1] where (0, 0) is center
            None if no face detected
        """
        results = self.model(frame, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return None

        # Select best face (highest confidence * area)
        boxes = results[0].boxes
        best_idx = 0
        best_score = 0

        h, w = frame.shape[:2]

        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1) / (w * h)
            score = conf * area

            if score > best_score:
                best_score = score
                best_idx = i

        if best_score == 0:
            return None

        # Get center of best box
        x1, y1, x2, y2 = boxes[best_idx].xyxy[0].tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Normalize to [-1, 1]
        norm_x = (cx / w) * 2 - 1
        norm_y = (cy / h) * 2 - 1

        return norm_x, norm_y
```

### Camera Worker with Face Tracking

```python
import threading
import time
from typing import Optional, Tuple
import numpy as np

class CameraWorker:
    """Threaded camera capture with face tracking."""

    TRACKING_TIMEOUT = 2.0  # seconds before returning to neutral
    INTERPOLATION_DURATION = 1.0  # seconds to interpolate back

    def __init__(self, robot: ReachyMini, tracker: Optional[YOLOHeadTracker] = None):
        self.robot = robot
        self.tracker = tracker

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        self._face_position: Optional[Tuple[float, float]] = None
        self._last_face_time: float = 0
        self._tracking_enabled = True

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _capture_loop(self):
        while self._running:
            frame = self.robot.media.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame.copy()

            # Face tracking
            if self.tracker and self._tracking_enabled:
                face_pos = self.tracker.detect(frame)
                if face_pos:
                    self._face_position = face_pos
                    self._last_face_time = time.monotonic()

            time.sleep(1/30)  # 30 Hz

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_face_tracking_offsets(self) -> Tuple[float, float, float, float, float, float]:
        """Get head offsets to look at detected face."""
        if not self._tracking_enabled or self._face_position is None:
            return 0, 0, 0, 0, 0, 0

        now = time.monotonic()
        time_since_face = now - self._last_face_time

        if time_since_face > self.TRACKING_TIMEOUT:
            # Interpolate back to neutral
            t = (time_since_face - self.TRACKING_TIMEOUT) / self.INTERPOLATION_DURATION
            if t >= 1.0:
                self._face_position = None
                return 0, 0, 0, 0, 0, 0
            factor = 1.0 - t
        else:
            factor = 1.0

        x, y = self._face_position

        # Convert to head movement (tuned for typical viewing distance)
        yaw = -x * 0.3 * factor  # radians, negative because looking left is positive x
        pitch = -y * 0.2 * factor  # radians

        return 0, 0, 0, 0, pitch, yaw

    def set_tracking_enabled(self, enabled: bool):
        self._tracking_enabled = enabled
        if not enabled:
            self._face_position = None
```

---

## Tool System

Dynamic tool dispatch for LLM function calling.

### Tool Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class ToolDependencies:
    """Dependencies injected into all tools."""
    robot: ReachyMini
    movement_manager: "MovementManager"
    camera_worker: Optional["CameraWorker"] = None
    vision_processor: Optional[Any] = None
    head_wobbler: Optional["HeadWobbler"] = None
    motion_duration: float = 1.0

class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for function calling."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description shown to LLM."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON schema for parameters."""
        pass

    @abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        pass

    def to_openai_spec(self) -> Dict[str, Any]:
        """Export as OpenAI function spec."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema
        }
```

### Example Tools

```python
class MoveHeadTool(Tool):
    @property
    def name(self) -> str:
        return "move_head"

    @property
    def description(self) -> str:
        return "Move the robot's head to look in a direction"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "up", "down", "front", "center"],
                    "description": "Direction to look"
                }
            },
            "required": ["direction"]
        }

    async def __call__(self, deps: ToolDependencies, direction: str) -> Dict[str, Any]:
        from reachy_mini.utils import create_head_pose

        poses = {
            "left": create_head_pose(yaw=30, degrees=True),
            "right": create_head_pose(yaw=-30, degrees=True),
            "up": create_head_pose(pitch=-20, degrees=True),
            "down": create_head_pose(pitch=20, degrees=True),
            "front": create_head_pose(),
            "center": create_head_pose(),
        }

        pose = poses.get(direction, poses["center"])

        # Queue the move
        from .moves import GotoQueueMove
        move = GotoQueueMove(
            target_pose=pose,
            duration=deps.motion_duration
        )
        deps.movement_manager.queue_move(move)

        return {"status": "ok", "direction": direction}


class DanceTool(Tool):
    DANCES = [
        "music1", "music2", "music3", "music4", "music5",
        "music6", "music7", "music8", "music9", "music10",
        "music11", "music12", "music13", "music14", "music15",
        "music16", "music17", "music18", "music19", "random"
    ]

    @property
    def name(self) -> str:
        return "dance"

    @property
    def description(self) -> str:
        return f"Make the robot dance. Available: {', '.join(self.DANCES)}"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dance_name": {
                    "type": "string",
                    "enum": self.DANCES,
                    "description": "Name of dance to perform"
                }
            },
            "required": ["dance_name"]
        }

    async def __call__(self, deps: ToolDependencies, dance_name: str) -> Dict[str, Any]:
        import random
        from reachy_mini.motion.recorded_move import RecordedMoves

        if dance_name == "random":
            dance_name = random.choice(self.DANCES[:-1])

        moves = RecordedMoves("pollen-robotics/reachy-mini-dances-library")
        move = moves.get(dance_name)
        deps.movement_manager.queue_move(move)

        return {"status": "ok", "dance": dance_name}


class CameraTool(Tool):
    @property
    def name(self) -> str:
        return "camera"

    @property
    def description(self) -> str:
        return "Take a photo and optionally ask a question about what you see"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to ask about the image"
                }
            },
            "required": []
        }

    async def __call__(self, deps: ToolDependencies, question: str = "") -> Dict[str, Any]:
        if deps.camera_worker is None:
            return {"error": "Camera not available"}

        frame = deps.camera_worker.get_latest_frame()
        if frame is None:
            return {"error": "No frame available"}

        if deps.vision_processor and question:
            import asyncio
            description = await asyncio.to_thread(
                deps.vision_processor.process_image,
                frame,
                question
            )
            return {"description": description}
        else:
            return {"status": "photo_taken"}


class HeadTrackingTool(Tool):
    @property
    def name(self) -> str:
        return "head_tracking"

    @property
    def description(self) -> str:
        return "Enable or disable automatic face tracking"

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "Whether to enable face tracking"
                }
            },
            "required": ["enabled"]
        }

    async def __call__(self, deps: ToolDependencies, enabled: bool) -> Dict[str, Any]:
        if deps.camera_worker:
            deps.camera_worker.set_tracking_enabled(enabled)
        return {"status": "ok", "tracking": enabled}
```

### Tool Registry

```python
import importlib
from pathlib import Path
from typing import Dict, List

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def all_specs(self) -> List[Dict[str, Any]]:
        return [t.to_openai_spec() for t in self._tools.values()]

    def load_from_profile(self, profile_path: Path):
        """Load tools listed in profile's tools.txt"""
        tools_file = profile_path / "tools.txt"
        if not tools_file.exists():
            return

        for line in tools_file.read_text().splitlines():
            tool_name = line.strip()
            if not tool_name or tool_name.startswith("#"):
                continue

            # Try profile-local first, then shared
            tool = self._load_tool(tool_name, profile_path)
            if tool:
                self.register(tool)

    def _load_tool(self, name: str, profile_path: Path) -> Optional[Tool]:
        # Try profile-local
        local_path = profile_path / f"{name}.py"
        if local_path.exists():
            spec = importlib.util.spec_from_file_location(name, local_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "tool"):
                return module.tool

        # Try shared tools
        try:
            module = importlib.import_module(f"tools.{name}")
            if hasattr(module, "tool"):
                return module.tool
        except ImportError:
            pass

        return None
```

---

## OpenAI Realtime Integration

Patterns for integrating with OpenAI's realtime API.

### Session Handler

```python
import asyncio
import json
from typing import Callable, Optional
from openai import AsyncOpenAI

class RealtimeHandler:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        tool_deps: ToolDependencies,
        on_transcript: Optional[Callable[[str, bool], None]] = None,
        on_audio_delta: Optional[Callable[[str], None]] = None
    ):
        self.client = AsyncOpenAI()
        self.tools = tool_registry
        self.deps = tool_deps
        self.on_transcript = on_transcript
        self.on_audio_delta = on_audio_delta

        self._connection = None
        self._connected = asyncio.Event()

    async def connect(self, instructions: str, voice: str = "cedar"):
        """Connect to realtime API."""
        self._connection = await self.client.realtime.connect(
            model="gpt-4o-realtime-preview"
        )

        await self._connection.session.update(
            instructions=instructions,
            voice=voice,
            tools=self.tools.all_specs(),
            input_audio_transcription={"model": "gpt-4o-transcribe"},
            turn_detection={"type": "server_vad"}
        )

        self._connected.set()

    async def run(self):
        """Main event loop."""
        async for event in self._connection:
            await self._handle_event(event)

    async def _handle_event(self, event):
        event_type = event.type

        if event_type == "response.audio.delta":
            if self.on_audio_delta:
                self.on_audio_delta(event.delta)

        elif event_type == "response.audio_transcript.delta":
            if self.on_transcript:
                self.on_transcript(event.delta, partial=True)

        elif event_type == "response.audio_transcript.done":
            if self.on_transcript:
                self.on_transcript(event.transcript, partial=False)

        elif event_type == "response.function_call_arguments.done":
            await self._handle_tool_call(event)

    async def _handle_tool_call(self, event):
        tool_name = event.name
        call_id = event.call_id

        try:
            args = json.loads(event.arguments)
        except json.JSONDecodeError:
            args = {}

        tool = self.tools.get(tool_name)
        if tool:
            result = await tool(self.deps, **args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        # Send result back
        await self._connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result)
            }
        )
        await self._connection.response.create()

    async def send_audio(self, audio_base64: str):
        """Send audio input."""
        await self._connection.input_audio_buffer.append(audio=audio_base64)

    async def update_personality(self, instructions: str, voice: str):
        """Update personality without reconnecting."""
        await self._connection.session.update(
            instructions=instructions,
            voice=voice
        )
```

---

## Profile System

Manage personalities, prompts, and tools.

### Profile Structure

```
profiles/
├── default/
│   ├── instructions.txt    # System prompt (supports [template] expansion)
│   ├── tools.txt           # List of tools to load
│   └── voice.txt           # Voice name (optional, defaults to "cedar")
├── example/
│   ├── instructions.txt
│   ├── tools.txt
│   └── sweep_look.py       # Custom tool for this profile
└── user_personalities/     # User-created profiles
    └── my_robot/
        ├── instructions.txt
        └── tools.txt
```

### Template Expansion

```python
from pathlib import Path
import re

def load_instructions(profile_path: Path, prompts_dir: Path) -> str:
    """Load instructions with template expansion."""
    instructions_file = profile_path / "instructions.txt"
    if not instructions_file.exists():
        return ""

    content = instructions_file.read_text()

    # Expand [path/to/template] placeholders
    def expand(match):
        template_path = prompts_dir / f"{match.group(1)}.txt"
        if template_path.exists():
            return template_path.read_text()
        return match.group(0)

    return re.sub(r'\[([^\]]+)\]', expand, content)

def load_voice(profile_path: Path) -> str:
    """Load voice from profile or default."""
    voice_file = profile_path / "voice.txt"
    if voice_file.exists():
        return voice_file.read_text().strip()
    return "cedar"
```

### Example instructions.txt

```
You are a helpful robot assistant named Reachy Mini.

[identities/friendly_helper]

[behaviors/conversational]

You have access to tools for moving your head, dancing, and using your camera.
When the user asks you to look at something, use the camera tool.
Express emotions through movement - dance when happy, look down when sad.
```

---

## Worker Thread Patterns

### Graceful Shutdown

```python
import threading
import signal
from typing import List

class WorkerManager:
    def __init__(self):
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._shutdown_event.set()

    def add_worker(self, target, name: str):
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._workers.append(thread)
        return thread

    def start_all(self):
        for worker in self._workers:
            worker.start()

    def wait_for_shutdown(self):
        self._shutdown_event.wait()

    def stop_all(self, timeout: float = 2.0):
        # Workers should check shutdown_event in their loops
        self._shutdown_event.set()
        for worker in self._workers:
            worker.join(timeout=timeout)
```

### Thread-Safe State

```python
from dataclasses import dataclass, field
from threading import Lock
from typing import Generic, TypeVar

T = TypeVar('T')

@dataclass
class ThreadSafeState(Generic[T]):
    """Thread-safe wrapper for mutable state."""
    _value: T
    _lock: Lock = field(default_factory=Lock)

    def get(self) -> T:
        with self._lock:
            return self._value

    def set(self, value: T):
        with self._lock:
            self._value = value

    def update(self, fn):
        with self._lock:
            self._value = fn(self._value)
            return self._value
```
