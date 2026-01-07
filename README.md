# Reachy Mini Skill for Claude Code

A Claude Code skill that provides complete SDK knowledge for controlling Reachy Mini robots.

## Features

- Head movement and antenna control
- Camera access and image processing
- Audio playback and recording
- Motion recording and playback
- Look-at functions (image and world coordinates)
- Deployment modes (USB, wireless, simulation, on-Pi)
- Advanced patterns: face tracking, LLM integration, OpenAI realtime

## Installation

### Add the marketplace

```bash
/plugin marketplace add gary149/reachy-mini-skill
```

### Install the skill

```bash
/plugin install reachy-mini@reachy-mini-marketplace
```

## Usage

Once installed, Claude Code will automatically use this skill when you ask about:

- Writing code to control Reachy Mini
- Moving the robot head or antennas
- Accessing camera/video
- Playing/recording audio
- Recording or playing back motions
- Looking at points in image or world space
- Connecting to real or simulated robot
- Building conversational AI apps with the robot
- Integrating with LLMs/OpenAI
- Deploying apps to the robot

## Quick Start Example

```python
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as robot:
    robot.wake_up()

    # Move head
    pose = create_head_pose(pitch=10, yaw=20, degrees=True)
    robot.goto_target(head=pose, antennas=[0.3, -0.3], duration=1.0)

    # Get camera frame
    frame = robot.media.get_frame()

    robot.goto_sleep()
```

## Documentation

- [SKILL.md](SKILL.md) - Main skill reference
- [Architecture & Deployment](references/architecture.md) - Daemon/client split, deployment modes
- [API Reference](references/api_reference.md) - Complete method signatures
- [Motion Reference](references/motion_reference.md) - Interpolation, Move classes
- [Media Reference](references/media_reference.md) - Camera resolutions, audio specs
- [Application Patterns](references/application_patterns.md) - Advanced patterns, LLM integration

## License

Apache-2.0
