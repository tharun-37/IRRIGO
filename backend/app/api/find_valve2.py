content = open('D:/Codings/MP3-V3/backend/app/api/compat.py', encoding='utf-8').read()
idx = content.find('@router.post("/devices/{device_id}/valve")')
print(content[idx:idx+500])