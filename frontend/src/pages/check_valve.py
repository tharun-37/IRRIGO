content = open('D:/Codings/MP3-V3/frontend/src/pages/FarmerView.tsx', encoding='utf-8').read()

# Step 1: Replace just the toggleValve function body
old_toggle = """  const toggleValve = () => {
    const deviceId = field.live?.deviceId ?? field.sensors.deviceId ?? ''
    if (!deviceId) return
    const isOpen = field.live?.valveOpen === true
    if (isOpen) {
      // Watering time not yet finished ? ask for confirmation
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

# The actual text has a replacement char, so let's search for it
idx = content.find('const toggleValve')
print('toggleValve at:', idx)
print(repr(content[idx:idx+100]))