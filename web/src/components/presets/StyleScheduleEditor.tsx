import {
  Box,
  Button,
  IconButton,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import AddIcon from '@mui/icons-material/Add'
import { useConfigStore } from '@/store/configStore'
import type { StyleScheduleEntry, RunningStyle } from '@/models/types'

const YEAR_OPTIONS = [
  { value: 1, label: 'Junior' },
  { value: 2, label: 'Classic' },
  { value: 3, label: 'Senior' },
] as const

const MONTH_OPTIONS = [
  { value: 1, label: 'Jan' },
  { value: 2, label: 'Feb' },
  { value: 3, label: 'Mar' },
  { value: 4, label: 'Apr' },
  { value: 5, label: 'May' },
  { value: 6, label: 'Jun' },
  { value: 7, label: 'Jul' },
  { value: 8, label: 'Aug' },
  { value: 9, label: 'Sep' },
  { value: 10, label: 'Oct' },
  { value: 11, label: 'Nov' },
  { value: 12, label: 'Dec' },
] as const

const HALF_OPTIONS = [
  { value: 1, label: 'Early' },
  { value: 2, label: 'Late' },
] as const

const STYLE_OPTIONS: { value: RunningStyle; label: string }[] = [
  { value: 'front', label: 'Front (Leader)' },
  { value: 'pace', label: 'Pace (Stalker)' },
  { value: 'late', label: 'Late (Betweener)' },
  { value: 'end', label: 'End (Chaser)' },
]

export default function StyleScheduleEditor({ presetId }: { presetId: string }) {
  const preset = useConfigStore((s) => s.getSelectedPreset().preset)
  const patchPreset = useConfigStore((s) => s.patchPreset)

  if (!preset) return null

  const schedule: StyleScheduleEntry[] = preset.styleSchedule ?? []

  const updateSchedule = (newSchedule: StyleScheduleEntry[]) => {
    patchPreset(presetId, 'styleSchedule', newSchedule)
  }

  const addEntry = () => {
    updateSchedule([
      ...schedule,
      { yearCode: 2, month: 6, half: 1, style: 'pace' },
    ])
  }

  const removeEntry = (index: number) => {
    updateSchedule(schedule.filter((_, i) => i !== index))
  }

  const updateEntry = (index: number, field: keyof StyleScheduleEntry, value: number | RunningStyle) => {
    const newSchedule = [...schedule]
    newSchedule[index] = { ...newSchedule[index], [field]: value }
    updateSchedule(newSchedule)
  }

  return (
    <Paper variant="outlined" sx={{ p: 1.5 }}>
      <Stack spacing={1.5}>
        <Typography variant="subtitle2">
          Style Schedule (changes running style at specified dates)
        </Typography>

        {schedule.length === 0 && (
          <Typography variant="body2" color="text.secondary">
            No scheduled style changes. Click "Add Style Change" to schedule one.
          </Typography>
        )}

        {schedule.map((entry, index) => (
          <Box key={index} sx={{ display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
            <Select
              size="small"
              value={entry.yearCode}
              onChange={(e) => updateEntry(index, 'yearCode', Number(e.target.value))}
              sx={{ minWidth: 90 }}
            >
              {YEAR_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Select
              size="small"
              value={entry.half}
              onChange={(e) => updateEntry(index, 'half', Number(e.target.value))}
              sx={{ minWidth: 70 }}
            >
              {HALF_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Select
              size="small"
              value={entry.month}
              onChange={(e) => updateEntry(index, 'month', Number(e.target.value))}
              sx={{ minWidth: 70 }}
            >
              {MONTH_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <Typography sx={{ mx: 0.5 }} color="text.secondary">→</Typography>

            <Select
              size="small"
              value={entry.style}
              onChange={(e) => updateEntry(index, 'style', e.target.value as RunningStyle)}
              sx={{ minWidth: 140 }}
            >
              {STYLE_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
              ))}
            </Select>

            <IconButton size="small" onClick={() => removeEntry(index)} color="error">
              <DeleteIcon fontSize="small" />
            </IconButton>
          </Box>
        ))}

        <Button
          size="small"
          startIcon={<AddIcon />}
          onClick={addEntry}
          sx={{ alignSelf: 'flex-start' }}
          variant="outlined"
        >
          Add Style Change
        </Button>
      </Stack>
    </Paper>
  )
}
