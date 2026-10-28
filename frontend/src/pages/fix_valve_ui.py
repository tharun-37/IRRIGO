import re

content = open('D:/Codings/MP3-V3/frontend/src/pages/FarmerView.tsx', encoding='utf-8').read()

old = """  const daysToStress = field.water.daysUntilStress

  const toggleValve = () => {
    const deviceId = field.live?.deviceId ?? field.sensors.deviceId ?? ''
    if (!deviceId) return
    const isOpen = field.live?.valveOpen === true
    if (isOpen) {
      // Watering time not yet finished — ask for confirmation
      const remaining = field.approval
        ? Math.max(0, field.approval.durationSeconds - (Date.now() / 1000 - new Date(field.approval.approvedAt).getTime() / 1000))
        : 0
      const remainingMin = Math.ceil(remaining / 60)
      const msg = remainingMin > 0
        ? `Watering time is not yet finished (${remainingMin} min remaining). Are you sure you want to close the valve?`
        : 'Are you sure you want to close the valve?'
      if (!window.confirm(msg)) return
    }
    const open = !isOpen
    api.valve(deviceId, open).catch(() => {})
    // Optimistic update
    if (field.live) {
      field.live.valveOpen = open
    }
  }"""

new = """  const daysToStress = field.water.daysUntilStress

  // Local valve state for optimistic updates
  const [valveOpen, setValveOpen] = useState<boolean>(field.live?.valveOpen === true)

  // Sync with server state when advisory refreshes
  useEffect(() => {
    if (field.live?.valveOpen != null) {
      setValveOpen(field.live.valveOpen)
    }
  }, [field.live?.valveOpen])

  const toggleValve = () => {
    const deviceId = field.live?.deviceId ?? field.sensors.deviceId ?? ''
    if (!deviceId) return
    const isOpen = valveOpen
    if (isOpen) {
      // Watering time not yet finished — ask for confirmation
      const remaining = field.approval
        ? Math.max(0, field.approval.durationSeconds - (Date.now() / 1000 - new Date(field.approval.approvedAt).getTime() / 1000))
        : 0
      const remainingMin = Math.ceil(remaining / 60)
      const msg = remainingMin > 0
        ? `Watering time is not yet finished (${remainingMin} min remaining). Are you sure you want to close the valve?`
        : 'Are you sure you want to close the valve?'
      if (!window.confirm(msg)) return
    }
    const open = !isOpen
    setValveOpen(open)  // Optimistic update
    api.valve(deviceId, open).catch(() => {
      // Revert on failure
      setValveOpen(isOpen)
    })
  }"""

assert old in content, 'Pattern not found'
content = content.replace(old, new)

# Also update the button text to use valveOpen instead of field.live?.valveOpen
old_button = """                  <CheckGlyph size={18} />
                  {field.live?.valveOpen ? 'Close valve' : 'Open valve'}"""
new_button = """                  <CheckGlyph size={18} />
                  {valveOpen ? 'Close valve' : 'Open valve'}"""
assert old_button in content, 'Button pattern not found'
content = content.replace(old_button, new_button)

# Update button color to use valveOpen
old_color = """                  className={`flex h-9 items-center justify-center gap-2 rounded-card border text-[13px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                    field.live?.valveOpen
                      ? 'border-red-600 bg-red-50 text-red-700 hover:border-red-700 hover:bg-red-100'
                      : 'border-status-healthy bg-status-healthySoft text-[#15803d] hover:border-[#15803d] hover:bg-green-100'
                  }`}"""
new_color = """                  className={`flex h-9 items-center justify-center gap-2 rounded-card border text-[13px] font-semibold transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                    valveOpen
                      ? 'border-red-600 bg-red-50 text-red-700 hover:border-red-700 hover:bg-red-100'
                      : 'border-status-healthy bg-status-healthySoft text-[#15803d] hover:border-[#15803d] hover:bg-green-100'
                  }`}"""
assert old_color in content, 'Color pattern not found'
content = content.replace(old_color, new_color)

open('D:/Codings/MP3-V3/frontend/src/pages/FarmerView.tsx', 'w', encoding='utf-8').write(content)
print('Updated FarmerView.tsx')