# Hardware

The controller is built around an ESP32 per irrigation zone. Each node reads one 7-in-1
soil probe, drives its own valve, and reports to the backend.

Firmware is not yet written. This document specifies what it must satisfy, so the hardware
can be commissioned and the firmware written against a fixed interface.

---

## The bench node, as built

One node exists today (Live-Field, `node-bench-01`). It is the closed-loop prototype the
production design above will be scaled from: hobby sensors, one servo-driven valve, and the
same actuation contract. Firmware lives outside the repository (Arduino sketch).

| Signal | Pin | Wiring notes |
| --- | --- | --- |
| Soil probe (analog) | GPIO 34 | input-only pin; calibrated `SOIL_RAW_WET` 1500 / `SOIL_RAW_DRY` 3400 |
| DHT11 data | GPIO 14 | input-only; pull-up belongs on the module, side effects on 35 make it read NaN |
| GY-30 light (BH1750) | SDA 33 / SCL 25 | second I²C bus (`Wire1`), address 0x23 |
| OLED SSD1306 | SDA 27 / SCL 26 | primary I²C bus (`Wire`), address 0x3C |
| Valve servo signal | GPIO 13 | ESP32Servo library, 500–2400 µs pulse, OPEN 45° / CLOSED 90° |

### Servo power — the one rule

The servo is powered **from an external 5 V rail with a common ground** to the ESP32, never
from the board's 3V3 pin. A servo rotating under load draws more than the regulator can
supply, and the brownout resets the board mid-irrigation — with the valve in whatever
position it died in.

### Node-side actuation limits

The closed loop acts on the backend's per-POST decision (`shouldIrrigate`,
`durationSeconds`). Because the backend sizes `durationSeconds` for a real pump — a 1000 m²
application can be computed at hours — the node enforces named, visible constants:

- `MAX_RUN_SECONDS = 900` — hard cap on one application, however large the backend asked.
- `REFRACTORY_MS = 120000` — minimum quiet time after a close before the valve may reopen.
- **Moisture hysteresis** — after an application closes, the valve stays closed until soil
  moisture has actually risen ~2 points above its level at close. A decision that stays
  true must not cycle the valve open/shut against a wet soil that has not yet registered.

The first two exist in the production fail-safe list below; the hysteresis is the bench
protection against decision/rate mismatch and should be carried into production firmware
with a field-calibrated threshold.

---

## Bill of materials, per zone

| Item | Purpose | Notes |
| --- | --- | --- |
| ESP32 DevKit V1 or WROOM-32 | Controller | 3.3 V logic, Wi-Fi |
| 7-in-1 soil probe | Moisture, temperature, EC, pH, N, P, K | RS485 variant, see below |
| RS485 transceiver (MAX3485 or SP3485) | Probe interface | 3.3 V, direction-controlled |
| 4-channel relay module | Valves and pump | See the isolation note below |
| Flow switch, pulse type | Actual water applied | Required for closed-loop accounting |
| 12 V power supply, 2 A minimum | Valves and pump | Separate from the logic rail |
| Buck converter to 3.3 V | Logic rail | Feed the ESP32 and transceiver only |
| Enclosure, IP65 | Field protection | Cable glands, drip loops |
| Solar panel and 18650, optional | Off-grid operation | Only with deep-sleep firmware |

## Power

The 12 V supply feeds the relay contacts and the buck converter. **Do not power the ESP32
3.3 V rail from the relay module's 5 V output**, and do not share a ground path between the
RS485 bus and the relay coil returns. Relay coil transients are the most common cause of
brownout resets and corrupted sensor readings in installations like this.

Keep the sensor cable on its own gauge, twisted where possible, with a shield grounded at
the controller end only.

---

## 7-in-1 soil probe

### Register map

The probe is addressed with Modbus RTU, function code `0x03` (read holding registers),
default `4800` baud, 8 data bits, no parity, 1 stop bit (8N1), slave address `0x01`.

| Register | Quantity | Value | Unit | Scale |
| --- | --- | --- | --- | --- |
| `0x0000` | 1 | Soil moisture | % volumetric | ÷ 100 |
| `0x0001` | 1 | Soil temperature | °C | ÷ 10, signed |
| `0x0002` | 1 | EC | µS/cm | ÷ 1 |
| `0x0003` | 1 | Soil pH | pH | ÷ 10 |
| `0x0004` | 1 | Nitrogen | mg/kg | ÷ 1 |
| `0x0005` | 1 | Phosphorus | mg/kg | ÷ 1 |
| `0x0006` | 1 | Potassium | mg/kg | ÷ 1 |

A read is one request for seven registers:

```
request   01 03 00 00 00 07 CRClo CRChi
response  01 03 0E <14 bytes of values> CRClo CRChi
```

### Verify before trusting

Probe vendors disagree on both scaling and register order, and several popular 7-in-1
boards ship with the registers in a different order from the table above. Before wiring a
node for real:

1. Read all seven registers with any RS485 tool and compare against the datasheet.
2. Confirm the moisture divisor. A raw value of `3500` means 35% only if the scale is ÷100.
3. Confirm temperature is signed and decimal, and that pH is ÷10.
4. Wet a beaker of distilled water and check pH reads near 7.0.
5. Dry the probe in air overnight and check moisture falls toward zero.

A probe that reports a plausible but wrong scale is far more dangerous than one that fails
loudly, because the model will act on the wrong number. The simulator in
`backend/tools/simulate_nodes.py` assumes the values above, so verify before comparing
against simulated data.

### Placement

Insert to the depth the crop's active roots reach, not just the surface. Keep the sensing
zone away from the wetted bulb edge, or readings will be dominated by a single irrigation
event rather than by root-zone storage. Take at least three readings across a zone before
treating any one of them as representative.

---

## Valves and pump

| Relay channel | Drives | Notes |
| --- | --- | --- |
| 1 | Zone valve | Solenoid, 12 V DC |
| 2 | Pump contactor | Where a pump serves several zones |
| 3 | Spare, or master leak valve | Fail-closed on controller loss |
| 4 | Spare | |

Use a latching or normally-closed arrangement so that a controller reset, a power cut or a
lost Wi-Fi connection results in a **closed** valve. A normally-open valve wired to a
relay that de-energises on fault will flood the field.

A solenoid valve coil draws current that will damage a small relay module over time. Above
roughly 1 A, drive the coil through a proper MOSFET or a rated relay, sized for inrush.

The flow switch is not optional if closed-loop accounting is to mean anything. Without it,
the system knows what it commanded but not what was delivered, and every water-saving
figure in the dashboard becomes an estimate of an estimate.

## Local fail-safes

Wiring alone is not enough. The firmware must implement:

- **Maximum runtime.** Close the valve after a configured limit regardless of backend
  state. This is the most important safety rule in the system.
- **Heartbeat.** If no successful backend contact within the interval, close the valve.
- **Watchdog** on the main loop, and a valve-off path that does not depend on it.
- **Minimum off time** between cycles, to stop pump short-cycling.

## Commissioning checklist

1. Confirm relay de-energises to valve-closed.
2. Confirm a backend outage closes the valve within the heartbeat interval.
3. Confirm the maximum-runtime rule fires with the backend unreachable.
4. Confirm the probe register map against the datasheet, using the checks above.
5. Compare a manual watering event against what the dashboard recorded.
6. Confirm the two ends of the RS485 bus carry a termination resistor, and only at the
   physical ends of the trunk.

## Safety

Mains voltage at the pump contactor must be handled by a qualified electrician. The
relay and ESP32 side is low voltage, but the two share an enclosure, so the enclosure
should be earthed and the low-voltage wiring kept physically separate from the mains
wiring inside it.
