# Feature Specification: Dynamic Running Style Schedule

## Overview

Enable users to configure automatic running style changes at specific dates during a training run. Currently, running style is only set once during the debut race via `juniorStyle`. This feature adds a schedule-based system where the style can change at any point in the career.

## Current Behavior

- `juniorStyle` field in preset config: `'end' | 'late' | 'pace' | 'front' | null`
- Style is only applied during debut race detection (`race_predebut` check in agent)
- `RaceFlow.set_strategy()` handles the UI interaction to select style
- Style selection requires `button_change` detection in pre-race lobby

## Proposed Behavior

Users can define a list of style change entries, each specifying:
- **Date**: Year + Month + Half (e.g., "Senior Early June")
- **Style**: The running style to use from that date onwards

The bot will:
1. Track the current active style based on the schedule
2. Apply the appropriate style at each race when `button_change` is available
3. Only trigger style selection when the style differs from what was last set

## Data Model

### Config Schema Changes

**File:** `web/src/models/config.schema.ts`

```typescript
// New schema for style schedule entry
export const styleScheduleEntrySchema = z.object({
  yearCode: z.number().min(0).max(4),  // 0=Pre-debut, 1=Junior, 2=Classic, 3=Senior, 4=Final
  month: z.number().min(1).max(12),
  half: z.number().min(1).max(2),      // 1=Early, 2=Late
  style: z.enum(['end', 'late', 'pace', 'front']),
});

export type StyleScheduleEntry = z.infer<typeof styleScheduleEntrySchema>;

// Add to preset schema
export const presetSchema = z.object({
  // ... existing fields ...
  juniorStyle: z.enum(['end', 'late', 'pace', 'front']).nullable(),  // Keep for debut
  styleSchedule: z.array(styleScheduleEntrySchema).default([]),       // NEW
});
```

### TypeScript Types

**File:** `web/src/models/types.ts`

```typescript
export interface StyleScheduleEntry {
  yearCode: number;   // 0-4
  month: number;      // 1-12
  half: number;       // 1=Early, 2=Late
  style: 'end' | 'late' | 'pace' | 'front';
}

export interface Preset {
  // ... existing fields ...
  juniorStyle?: 'end' | 'late' | 'pace' | 'front' | null;
  styleSchedule?: StyleScheduleEntry[];
}
```

### Python Settings

**File:** `core/settings.py`

```python
# In extract_runtime_preset()
style_schedule = preset.get("styleSchedule", [])
# Returns: List[Dict] with keys: yearCode, month, half, style
```

## Backend Implementation

### Style Schedule Manager

**New file:** `core/utils/style_schedule.py`

```python
from dataclasses import dataclass
from typing import Optional, List
from core.utils.date_uma import DateInfo, date_cmp

@dataclass
class StyleScheduleEntry:
    year_code: int
    month: int
    half: int
    style: str  # 'end' | 'late' | 'pace' | 'front'

    def as_date_info(self) -> DateInfo:
        return DateInfo(
            raw=f"Y{self.year_code}-{self.month}-{self.half}",
            year_code=self.year_code,
            month=self.month,
            half=self.half,
        )

class StyleScheduleManager:
    def __init__(self, schedule: List[dict], debut_style: Optional[str] = None):
        self.debut_style = debut_style
        self.entries = sorted(
            [StyleScheduleEntry(**e) for e in schedule],
            key=lambda e: (e.year_code, e.month, e.half)
        )
        self._last_applied_style: Optional[str] = None

    def get_style_for_date(self, date: DateInfo) -> Optional[str]:
        """
        Returns the style that should be active at the given date.
        Finds the latest schedule entry that is <= the current date.
        """
        active_style = self.debut_style

        for entry in self.entries:
            entry_date = entry.as_date_info()
            if date_cmp(entry_date, date) <= 0:
                active_style = entry.style
            else:
                break  # entries are sorted, no need to continue

        return active_style

    def should_apply_style(self, date: DateInfo) -> tuple[bool, Optional[str]]:
        """
        Returns (should_apply, style) based on current date.
        Only returns True if the style differs from last applied.
        """
        style = self.get_style_for_date(date)
        if style and style != self._last_applied_style:
            return True, style
        return False, None

    def mark_applied(self, style: str):
        """Call after successfully applying a style."""
        self._last_applied_style = style

    def reset(self):
        """Reset tracking (e.g., at career start)."""
        self._last_applied_style = None
```

### Agent Integration

**Files:** `core/actions/ura/agent.py`, `core/actions/unity_cup/agent.py`

```python
from core.utils.style_schedule import StyleScheduleManager

class AgentURA:
    def __init__(self, ..., select_style=None, style_schedule=None, ...):
        self.style_manager = StyleScheduleManager(
            schedule=style_schedule or [],
            debut_style=select_style
        )

    # In race handling logic (around line 375-425):
    def _get_race_style(self, current_date: DateInfo, is_debut: bool) -> Optional[str]:
        if is_debut:
            # Debut always uses debut_style if set
            return self.style_manager.debut_style

        should_apply, style = self.style_manager.should_apply_style(current_date)
        if should_apply:
            return style
        return None

    # In race execution:
    def handle_race(self, ...):
        # ... existing date detection ...

        style_to_apply = self._get_race_style(current_date, race_predebut)

        if style_to_apply:
            ok = self.race.run(
                select_style=style_to_apply,
                # ... other params ...
            )
            if ok:
                self.style_manager.mark_applied(style_to_apply)
```

### Settings Extraction

**File:** `core/settings.py`

```python
# In extract_runtime_preset() around line 760
return {
    # ... existing fields ...
    "select_style": select_style,
    "style_schedule": preset.get("styleSchedule", []),  # NEW
}
```

### Main.py Integration

**File:** `main.py`

```python
# Around line 283-325
self.agent_scenario = AgentURA(
    # ... existing params ...
    select_style=preset_opts["select_style"],
    style_schedule=preset_opts["style_schedule"],  # NEW
)
```

## Frontend Implementation

### Style Schedule Editor Component

**New file:** `web/src/components/presets/StyleScheduleEditor.tsx`

```tsx
import { Paper, Stack, Typography, Button, IconButton, Select, MenuItem, Box } from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import { useConfigStore } from '../../store/configStore';
import { StyleScheduleEntry } from '../../models/types';

const YEAR_OPTIONS = [
  { value: 1, label: 'Junior' },
  { value: 2, label: 'Classic' },
  { value: 3, label: 'Senior' },
];

const MONTH_OPTIONS = Array.from({ length: 12 }, (_, i) => ({
  value: i + 1,
  label: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][i],
}));

const HALF_OPTIONS = [
  { value: 1, label: 'Early' },
  { value: 2, label: 'Late' },
];

const STYLE_OPTIONS = [
  { value: 'front', label: 'Front (逃げ)' },
  { value: 'pace', label: 'Pace (先行)' },
  { value: 'late', label: 'Late (差し)' },
  { value: 'end', label: 'End (追込)' },
];

export default function StyleScheduleEditor({ presetId }: { presetId: string }) {
  const preset = useConfigStore((s) => s.getSelectedPreset().preset);
  const patchPreset = useConfigStore((s) => s.patchPreset);

  if (!preset) return null;

  const schedule = preset.styleSchedule ?? [];

  const updateSchedule = (newSchedule: StyleScheduleEntry[]) => {
    patchPreset(presetId, 'styleSchedule', newSchedule);
  };

  const addEntry = () => {
    updateSchedule([
      ...schedule,
      { yearCode: 2, month: 6, half: 1, style: 'pace' },
    ]);
  };

  const removeEntry = (index: number) => {
    updateSchedule(schedule.filter((_, i) => i !== index));
  };

  const updateEntry = (index: number, field: keyof StyleScheduleEntry, value: any) => {
    const newSchedule = [...schedule];
    newSchedule[index] = { ...newSchedule[index], [field]: value };
    updateSchedule(newSchedule);
  };

  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Stack spacing={1.5}>
        <Typography variant="subtitle2">
          Style Schedule (changes style at specified dates)
        </Typography>

        {schedule.map((entry, index) => (
          <Box key={index} sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <Select
              size="small"
              value={entry.yearCode}
              onChange={(e) => updateEntry(index, 'yearCode', e.target.value)}
              sx={{ width: 100 }}
            >
              {YEAR_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Select
              size="small"
              value={entry.half}
              onChange={(e) => updateEntry(index, 'half', e.target.value)}
              sx={{ width: 80 }}
            >
              {HALF_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Select
              size="small"
              value={entry.month}
              onChange={(e) => updateEntry(index, 'month', e.target.value)}
              sx={{ width: 80 }}
            >
              {MONTH_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Typography sx={{ mx: 1 }}>→</Typography>

            <Select
              size="small"
              value={entry.style}
              onChange={(e) => updateEntry(index, 'style', e.target.value)}
              sx={{ width: 140 }}
            >
              {STYLE_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <IconButton size="small" onClick={() => removeEntry(index)}>
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Box>
        ))}

        <Button
          size="small"
          startIcon={<AddIcon />}
          onClick={addEntry}
          sx={{ alignSelf: 'flex-start' }}
        >
          Add Style Change
        </Button>
      </Stack>
    </Paper>
  );
}
```

### Integration in PresetPanel

**File:** `web/src/components/presets/PresetPanel.tsx`

Add the StyleScheduleEditor below the existing StyleSelector:

```tsx
import StyleScheduleEditor from './StyleScheduleEditor';

// In the render, after StyleSelector:
<StyleSelector presetId={active.id!} />
<StyleScheduleEditor presetId={active.id!} />
```

## Migration

### Config Store Migration

**File:** `web/src/store/configStore.ts`

```typescript
// In replaceConfig/importJson migrations:
const migrated = {
  ...config,
  presets: config.presets.map(preset => ({
    ...preset,
    styleSchedule: preset.styleSchedule ?? [],  // Default empty array
  })),
};
```

## Testing

### Unit Tests

**File:** `tests/core/utils/test_style_schedule.py`

```python
import pytest
from core.utils.style_schedule import StyleScheduleManager, StyleScheduleEntry
from core.utils.date_uma import DateInfo

def test_empty_schedule_returns_debut_style():
    mgr = StyleScheduleManager(schedule=[], debut_style='front')
    date = DateInfo(raw='', year_code=2, month=6, half=1)
    assert mgr.get_style_for_date(date) == 'front'

def test_schedule_overrides_at_correct_date():
    mgr = StyleScheduleManager(
        schedule=[
            {'yearCode': 2, 'month': 6, 'half': 1, 'style': 'late'},
        ],
        debut_style='front'
    )

    # Before schedule date
    before = DateInfo(raw='', year_code=2, month=5, half=2)
    assert mgr.get_style_for_date(before) == 'front'

    # At schedule date
    at = DateInfo(raw='', year_code=2, month=6, half=1)
    assert mgr.get_style_for_date(at) == 'late'

    # After schedule date
    after = DateInfo(raw='', year_code=3, month=1, half=1)
    assert mgr.get_style_for_date(after) == 'late'

def test_multiple_schedule_entries():
    mgr = StyleScheduleManager(
        schedule=[
            {'yearCode': 2, 'month': 6, 'half': 1, 'style': 'pace'},
            {'yearCode': 3, 'month': 1, 'half': 1, 'style': 'end'},
        ],
        debut_style='front'
    )

    junior = DateInfo(raw='', year_code=1, month=12, half=2)
    assert mgr.get_style_for_date(junior) == 'front'

    classic_late = DateInfo(raw='', year_code=2, month=9, half=1)
    assert mgr.get_style_for_date(classic_late) == 'pace'

    senior = DateInfo(raw='', year_code=3, month=6, half=1)
    assert mgr.get_style_for_date(senior) == 'end'

def test_should_apply_tracks_last_applied():
    mgr = StyleScheduleManager(
        schedule=[{'yearCode': 2, 'month': 6, 'half': 1, 'style': 'late'}],
        debut_style='front'
    )

    date = DateInfo(raw='', year_code=2, month=6, half=1)

    # First check should want to apply
    should, style = mgr.should_apply_style(date)
    assert should is True
    assert style == 'late'

    # Mark as applied
    mgr.mark_applied('late')

    # Second check should not want to apply (already applied)
    should, style = mgr.should_apply_style(date)
    assert should is False
```

### Web UI Tests

**File:** `tests/web/test_config_schema.py`

Add tests for styleSchedule validation:

```typescript
// Test that styleSchedule defaults to empty array
// Test that invalid entries are rejected
// Test migration from configs without styleSchedule
```

## Acceptance Criteria

1. **UI**: Users can add/remove/edit style schedule entries in the Presets panel
2. **Persistence**: Style schedule saves to config.json and loads correctly
3. **Runtime**: Bot applies scheduled style changes at the correct dates
4. **Debut**: Existing `juniorStyle` behavior preserved for debut race
5. **Efficiency**: Style only applied when it changes (no redundant clicks)
6. **Migration**: Existing configs without styleSchedule work (default empty)

## File Changes Summary

| File | Change |
|------|--------|
| `web/src/models/config.schema.ts` | Add `styleScheduleEntrySchema`, add `styleSchedule` to preset |
| `web/src/models/types.ts` | Add `StyleScheduleEntry` interface |
| `web/src/store/configStore.ts` | Add migration for `styleSchedule` default |
| `web/src/components/presets/StyleScheduleEditor.tsx` | **NEW** - UI component |
| `web/src/components/presets/PresetPanel.tsx` | Import and render StyleScheduleEditor |
| `core/utils/style_schedule.py` | **NEW** - StyleScheduleManager class |
| `core/settings.py` | Extract `styleSchedule` in `extract_runtime_preset()` |
| `core/actions/ura/agent.py` | Integrate StyleScheduleManager |
| `core/actions/unity_cup/agent.py` | Integrate StyleScheduleManager |
| `main.py` | Pass `style_schedule` to agent constructor |
| `tests/core/utils/test_style_schedule.py` | **NEW** - Unit tests |

## Open Questions

1. Should style changes also apply to Team Trials / Daily Races (AgentNav flows)?
2. Should there be validation to prevent duplicate date entries?
3. Should the UI show a visual timeline of style changes?
