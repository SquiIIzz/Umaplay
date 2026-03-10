# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Umaplay is an AI-driven automation bot for *Umamusume: Pretty Derby* that automates training, races, and skill management. It uses YOLO object detection, OCR (PaddleOCR), logistic regression classifiers, and custom scoring logic to play the game automatically on Steam (PC) or Android (via scrcpy/ADB).

## Build & Run Commands

### Python Backend (Bot Runtime)
```bash
# Create and activate environment
conda create -n env_uma python==3.10
conda activate env_uma
pip install -r requirements.txt

# Run the bot (starts bot + config server at http://127.0.0.1:8000)
python main.py
python main.py --port 8080  # Custom port

# Run remote inference server (for offloading heavy processing)
uvicorn server.main_inference:app --host 0.0.0.0 --port 8001
```

### Web UI (React/Vite)
```bash
cd web
npm install
npm run dev      # Development server
npm run build    # Production build (outputs to web/dist/)
npm run lint     # ESLint
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_turns.py

# Run tests matching pattern
pytest tests/core/test_settings.py -k scenario
```

## Architecture

### Runtime Core Loop
1. **Capture** game window via controller (Steam/Scrcpy/BlueStacks/ADB)
2. **Perceive** frame with YOLO detectors and OCR pipelines
3. **Assemble state** using analyzers and extractors
4. **Decide** next action in agent loop (`core/agent.py`)
5. **Act** through controller clicks, synchronized via `Waiter`
6. **Expose** configuration through FastAPI server

### Key Directories
```
core/
├── actions/           # Automation flows (training, races, events, skills)
│   ├── ura/          # URA scenario-specific flows
│   └── unity_cup/    # Unity Cup scenario-specific flows
├── controllers/       # Platform abstractions (Steam, Scrcpy, ADB)
├── perception/        # YOLO, OCR, analyzers, extractors
│   └── analyzers/    # Screen classifiers, hint detection, template matching
├── scenarios/         # Scenario registry and policy routing
├── utils/            # Waiter, logger, skill memory, event processing
└── settings.py       # Runtime configuration mapping

server/
├── main.py           # FastAPI config server
└── main_inference.py # Remote perception service

web/src/
├── components/       # React components (general/, presets/, events/, nav/)
├── models/           # Zod schemas and TypeScript types
├── store/            # Zustand state management
└── services/         # API client

datasets/in_game/     # Skills, races, events JSON catalogs
models/               # YOLO weights, classifiers
prefs/                # Runtime config (config.json)
```

### Scenario System
Scenarios (URA, Unity Cup) are registered in `core/scenarios/registry.py`. Each scenario has:
- Policy functions for training decisions
- Scenario-specific flows in `core/actions/<scenario>/`
- UI strategy components in `web/src/components/presets/strategy/`

To add a new scenario, follow `docs/ai/SOPs/adding-new-scenario.md`.

### Configuration Flow
1. Web UI uses Zod schemas (`web/src/models/config.schema.ts`)
2. Zustand store (`web/src/store/configStore.ts`) manages state
3. FastAPI persists to `prefs/config.json`
4. Python `Settings.apply_config()` maps to runtime constants

For config changes spanning frontend/backend, follow `docs/ai/SOPs/sop-config-back-front.md`.

### Waiter Pattern
`core/utils/waiter.py` orchestrates detection-driven clicks and polling:
- `click_when()` - Wait for detection then click
- `seen()` - Single snapshot probe
- Use `forbid_texts` to prevent misclicks via OCR validation

See `docs/ai/SOPs/waiter-usage-and-integration.md` for integration patterns.

## Key Patterns

### Button Activation Detection
```python
from core.perception.is_button_active import ActiveButtonClassifier
clf = ActiveButtonClassifier.load(Settings.IS_BUTTON_ACTIVE_CLF_PATH)
```

### Skill Matching
`core/utils/skill_matching.py` normalizes OCR text with rule-based token checks. Manual overrides in `datasets/in_game/skill_matching_overrides.json` handle ambiguous skills requiring `require_tokens`/`forbid_tokens`.

### Debug Artifacts
Low-confidence YOLO/OCR snapshots save to `debug/<agent>/<tag>/raw`. Enable via `Settings.DEBUG`.

## Development Workflow

### Notebook Prototyping
When asked to prototype, use `dev_nav.ipynb` or `dev_play.ipynb`. Place code in `## PROTOTYPE` sections with destination module comments. Wait for explicit approval before migrating to `.py` files.

### Contributing
- Fork repo, create branch, PR into **dev** branch
- Follow SOLID principles and clean architecture
- Read relevant SOPs in `docs/ai/SOPs/` before making structural changes

## Hotkeys (Runtime)
- **F2**: Start/stop training bot
- **F7**: Team Trials automation
- **F8**: Daily Races automation
- **F9**: Roulette/Prize Derby

## Important SOPs
- `docs/ai/SYSTEM_OVERVIEW.md` - Full architecture documentation
- `docs/ai/SOPs/waiter-usage-and-integration.md` - Waiter pattern usage
- `docs/ai/SOPs/adding-new-scenario.md` - Adding new game scenarios
- `docs/ai/SOPs/sop-config-back-front.md` - Frontend/backend config sync
