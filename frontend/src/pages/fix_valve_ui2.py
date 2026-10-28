content = open('D:/Codings/MP3-V3/frontend/src/pages/FarmerView.tsx', encoding='utf-8').read()

# Find toggleValve function
start = content.find('  const toggleValve = () => {')
end = content.find('  const plots = useMemo', start)

# Replace the entire toggleValve function
new_func = '''  // Local valve state for optimistic updates
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
      // Watering time not yet finished - ask for confirmation
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
    setValveOpen(open)
    api.valve(deviceId, open).catch(() => {
      setValveOpen(isOpen)
    })
  }

'''

content = content[:start] + new_func + content[end:]

# Also update the button text and color to use valveOpen instead of field.live?.valveOpen
content = content.replace("field.live?.valveOpen ? 'Close valve' : 'Open valve'", "valveOpen ? 'Close valve' : 'Open valve'")
content = content.replace("field.live?.valveOpen\n                      ? 'border-red-600", "valveOpen\n                      ? 'border-red-600")

open('D:/Codings/MP3-V3/frontend/src/pages/FarmerView.tsx', 'w', encoding='utf-8').write(content)
print('Updated FarmerView.tsx')