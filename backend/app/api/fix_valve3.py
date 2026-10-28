content = open('D:/Codings/MP3-V3/backend/app/api/compat.py', encoding='utf-8').read()

old = '''@router.post("/devices/{device_id}/valve")
def valve(
    request: Request,
    device_id: str,
    body: dict[str, Any] | None = None,
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    try:
        compat.device(device_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}") from error
    open_ = bool((body or {}).get("open", False))
    from ..db.repository import ManualValveCommandRepository
    from ..core.database import get_database
    repo = ManualValveCommandRepository(get_database())
    repo.create(device_id, open_)
    EventRepository(get_database()).record(
        kind="valve-manual",
        device_id=device_id,
        detail=f"Manual valve command: {'OPEN' if open_ else 'CLOSE'}",
        source="dashboard",
    )
    return {"accepted": True, "deviceId": device_id, "valveOpen": open_}'''

new = '''@router.post("/devices/{device_id}/valve")
def valve(
    request: Request,
    device_id: str,
    body: dict[str, Any] | None = None,
    compat: CompatService = Depends(get_compat),
) -> dict[str, Any]:
    # Look up the device from the database (not the compat registry),
    # so bench nodes that have posted telemetry are accepted.
    from ..db.repository import DeviceRepository
    from ..core.database import get_database
    device_repo = DeviceRepository(get_database())
    device = None
    try:
        device = device_repo.get(device_id)
    except KeyError:
        pass
    if device is None:
        with get_database().cursor() as cursor:
            cursor.execute("SELECT * FROM devices WHERE id = ?", (device_id,))
            row = cursor.fetchone()
            if row is not None:
                device = {"id": row["id"]}
    if device is None:
        raise HTTPException(status_code=404, detail=f"no device {device_id!r}")
    open_ = bool((body or {}).get("open", False))
    from ..db.repository import ManualValveCommandRepository
    repo = ManualValveCommandRepository(get_database())
    repo.create(device_id, open_)
    EventRepository(get_database()).record(
        kind="valve-manual",
        device_id=device_id,
        detail=f"Manual valve command: {'OPEN' if open_ else 'CLOSE'}",
        source="dashboard",
    )
    return {"accepted": True, "deviceId": device_id, "valveOpen": open_}'''

assert old in content, 'Pattern not found'
content = content.replace(old, new)
open('D:/Codings/MP3-V3/backend/app/api/compat.py', 'w', encoding='utf-8').write(content)
print('Fixed valve endpoint')