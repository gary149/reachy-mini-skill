# Architecture & Deployment

How the Reachy Mini software stack works and how to run code on the robot.

## Table of Contents
- [Client-Server Architecture](#client-server-architecture)
- [Deployment Modes](#deployment-modes)
- [Running Your Code](#running-your-code)
- [App Distribution System](#app-distribution-system)

---

## Client-Server Architecture

Reachy Mini uses a **daemon (server) + SDK (client)** split:

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Computer                           │
│                                                             │
│  ┌─────────────────┐         ┌─────────────────────────┐   │
│  │  Your Python    │  HTTP/  │       Daemon            │   │
│  │  Code (SDK)     │──WS────►│  - REST API :8000       │   │
│  │                 │         │  - Hardware I/O         │   │
│  │  ReachyMini()   │         │  - Safety checks        │   │
│  └─────────────────┘         └───────────┬─────────────┘   │
│                                          │ USB/Serial      │
└──────────────────────────────────────────┼─────────────────┘
                                           ▼
                                    ┌──────────────┐
                                    │  Reachy Mini │
                                    │   Hardware   │
                                    └──────────────┘
```

**Daemon responsibilities:**
- Communicates with motors via USB/Serial
- Runs safety checks and limits
- Exposes REST API (localhost:8000) and WebSocket
- Manages camera and audio hardware

**SDK responsibilities:**
- Your Python code
- Connects to daemon over network
- Provides high-level API (`goto_target`, `look_at_image`, etc.)

**Key insight:** The robot hardware has no software - it's just motors and sensors. The daemon is what controls it.

---

## Deployment Modes

### 1. Simulation (No Robot)

Test code without hardware using MuJoCo physics simulation.

```bash
# Terminal 1: Start daemon in simulation mode
reachy-mini-daemon --sim
```

```python
# Your code
robot = ReachyMini(use_sim=True)
```

### 2. USB / Lite (Robot on Your Computer)

Robot connected directly to your computer via USB cable.

```bash
# Terminal 1: Start the daemon
reachy-mini-daemon
```

```python
# Terminal 2: Your code (connects to localhost)
robot = ReachyMini()  # localhost_only=True by default
```

### 3. Wireless - Remote Laptop

Daemon runs on Raspberry Pi inside the robot. Your code runs on your laptop.

```
┌────────────────────┐       WiFi        ┌─────────────────────┐
│  Your Laptop       │◄─────────────────►│  Raspberry Pi       │
│                    │  WebRTC streams   │  (inside robot)     │
│  Your Python code  │                   │  reachy-mini-daemon │
└────────────────────┘                   └─────────────────────┘
```

```python
# On your laptop
robot = ReachyMini(localhost_only=False, media_backend="webrtc")
```

### 4. Wireless - 100% On Pi

Everything runs on the Raspberry Pi inside the robot. No laptop needed.

```
┌─────────────────────────────────────────┐
│           Raspberry Pi                  │
│  ┌─────────────────────────────────┐    │
│  │  Your App                       │    │
│  │  ReachyMini(media_backend=      │    │
│  │            "gstreamer")         │    │
│  └──────────────┬──────────────────┘    │
│                 │ localhost             │
│  ┌──────────────▼──────────────────┐    │
│  │  reachy-mini-daemon             │    │
│  └──────────────┬──────────────────┘    │
│                 │ USB                   │
└─────────────────┼───────────────────────┘
                  ▼
           ┌──────────────┐
           │  Reachy Mini │
           └──────────────┘
```

```python
# On the Pi
robot = ReachyMini(media_backend="gstreamer")
```

### Summary Table

| Mode | Daemon runs on | SDK runs on | Connection |
|------|----------------|-------------|------------|
| Simulation | Your computer | Your computer | localhost |
| USB/Lite | Your computer | Your computer | localhost |
| Wireless (remote) | Pi | Your laptop | WiFi/WebRTC |
| Wireless (on-device) | Pi | Pi | localhost/GStreamer |

---

## Running Your Code

### Step 1: Start the Daemon

The daemon must be running before your code can connect.

```bash
# Simulation
reachy-mini-daemon --sim

# Real robot (USB)
reachy-mini-daemon

# Real robot (wireless) - daemon usually runs as a service on Pi
# Already running, or: reachy-mini-daemon --wireless-version
```

### Step 2: Run Your Python Script

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as robot:
    robot.wake_up()

    pose = create_head_pose(yaw=20, pitch=10, degrees=True)
    robot.goto_target(head=pose, duration=1.0)

    robot.goto_sleep()
```

### Auto-Spawn Daemon

For convenience, you can have the SDK spawn the daemon automatically:

```python
# Spawns daemon if not already running
robot = ReachyMini(spawn_daemon=True)
```

### Connection Parameters

```python
ReachyMini(
    localhost_only=True,      # True: localhost only, False: network discovery
    spawn_daemon=False,       # Auto-spawn daemon if not running
    use_sim=False,            # Connect to simulation daemon
    timeout=5.0,              # Connection timeout (seconds)
    media_backend="default",  # "default", "gstreamer", "webrtc", "no_media"
)
```

---

## App Distribution System

For distributing complete applications (not just scripts), Reachy Mini has an app system.

### Creating an App

```bash
# Interactive scaffolding
python -m reachy_mini.apps.app create
```

Generates:
```
my_app/
├── my_app/
│   ├── __init__.py
│   ├── main.py           # ReachyMiniApp subclass
│   └── static/           # Optional settings UI
├── pyproject.toml        # Entry point registration
└── README.md             # HuggingFace metadata
```

### ReachyMiniApp Base Class

```python
from reachy_mini import ReachyMini, ReachyMiniApp
import threading

class MyApp(ReachyMiniApp):
    # Optional: Settings UI URL (shows icon in dashboard)
    custom_app_url = "http://0.0.0.0:7860/"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Main app logic. Robot already connected."""

        while not stop_event.is_set():
            # Your logic here
            reachy_mini.goto_target(...)

            if stop_event.wait(timeout=0.1):
                break
```

### Register Entry Point

In `pyproject.toml`:

```toml
[project.entry-points."reachy_mini_apps"]
my_app = "my_app.main:MyApp"
```

### Validate and Publish

```bash
# Check app structure
python -m reachy_mini.apps.app check

# Publish to HuggingFace Spaces
python -m reachy_mini.apps.app publish

# Request official listing
python -m reachy_mini.apps.app publish --official
```

### README.md Metadata

Required for HuggingFace discovery:

```yaml
---
title: My App
tags:
  - reachy_mini
  - reachy_mini_python_app
short_description: Does something cool
---
```

### Managing Apps

**Dashboard** (localhost:8000):
- Browse available apps
- Install from HuggingFace Spaces
- Start/stop apps
- View logs

**App Lifecycle:**
- Apps run as subprocesses
- Daemon manages start/stop
- `stop_event` signals graceful shutdown
- Robot returns to zero position after app stops

### App Sources

| Source | Description |
|--------|-------------|
| HuggingFace Spaces | Tagged with `reachy_mini_python_app` |
| Official Store | Curated list from Pollen Robotics |
| Installed | Already installed locally |
| Local | Filesystem path for development |
