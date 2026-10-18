# AURA-1 — Window-Replacement Reversible Heat Pump
## Complete Engineering Reference for `AC.py`

`AC.py` is a single-file, physics-level **working replica** of the machine
specified in `ProjectGoal.md`: a window-replacing, season-reversible,
ultra-quiet vapor-compression heat pump. Every subsystem — real-gas
refrigerant properties (Peng–Robinson EOS), inverter compressor,
micro-channel coils, ECM fans, expansion train (EEV + ejector + IHX +
staged EVI + liquid quench), reversing mechanism (valve or physical
180° flip), frost/defrost, condensate atomization with evaporative
assist, acoustics, controls, installation envelope, and the room it
conditions — is implemented and solved numerically.

**Pure stdlib** (`math`, `argparse`, `dataclasses`, `time`, `json`,
`contextlib`, `io`) + `pyglet`/`PyOpenGL` for the viewer only. Python ≥3.9.
No numpy, no CoolProp — every property function is implemented inline.

| Command | Result |
|---|---|
| `python AC.py` | launches the 5-tab interactive dashboard |
| `python AC.py --validate` | proof suite (conservation, Carnot, properties) |
| `python AC.py --test` | 15-check self-test |
| `python AC.py --blueprint` | writes `AURA-1_blueprint.json` |
| `python AC.py --all` | everything (console modes, then viewer) |

---

## 1. Repository Layout

| File | Role |
|---|---|
| `AC.py` | the entire model, viewer, harness, and CLI (~2900 lines) |
| `overview.md` | this document |
| `ProjectGoal.md` | the original product specification |
| `AURA-1_diagram.svg` | generated to-scale engineering sheet (`--svg`) |
| `AURA-1_blueprint.json` | generated frozen design package (`--blueprint`) |
| `ReferenceCode/` | supplied reference implementations (tab patterns, GL idioms) |

## 2. Design Targets vs. Modeled Results

| Metric | Target | Typical window AC | AURA-1 modeled |
|---|---|---|---|
| Cooling COP @ 95 °F | > 4.5 | ~3.5 (CEER 12) | **4.86** @ 27.4 dBA |
| Cooling COP @ 82 °F part-load | — | — | **7.26** @ 18.1 dBA |
| Heating COP @ 47 °F | > 4.5 | 1.0 (resistance) | **4.61** @ 26.2 dBA |
| Seasonal cooling | SEER > 20 | ~12 | **≈22.4** (bin est.) |
| Seasonal heating | HSPF > 13 | n/a | **≈12.6** (bin est.) |
| Heating COP @ 5 °F | > 2.5 | n/a | **3.16** (EVI + quench) |
| Heating @ −50 °C | feasible | none exist | **1.5 kW @ COP 1.82** (EVI×3) |
| Indoor noise | < 40 dBA max | 50–60 dBA | **18–36 dBA; 25 dBA night cap** |
| Condensate | zero drip | drips | atomized + evaporative assist |
| Infiltration | near-zero | 30–50 W/K equiv. | **6 W/K** |
| Mode change | effortless | manual seasonal swap | valve *or* 180° flip |

Numbers hardened by the real-gas EOS pass plus the loss stack (part-load
motor derate, ṁ² line Δp, shell loss, maldistribution). Earlier
correlation-only numbers ran ~15–25% optimistic; the current values are
the honest ones.

## 3. Code Map (§ blocks inside `AC.py`)

| § | Lines of responsibility | Marker / symbol |
|---|---|---|
| 1 | Constants, K/C/F helpers, dB summation, dataclasses | `db_sum`, `OperatingPoint`, `CycleResult` |
| 2 | R-32 properties: Wagner psat, Watson hfg, cp fits, **Peng–Robinson EOS** (Cardano Z-solve, h/s departures, isentrope inversions) | `class RefrigerantR32`, `z_vapor`, `h_vap`, `s_vap`, `_solve_t_for_s`, `_solve_t_for_h` |
| 3 | Psychrometrics — Magnus p_ws, humidity ratio, condensate split | `p_ws`, `w_ratio`, `condensate_rate` |
| 4 | Inverter twin-rotary compressor, real-gas suction density, part-load η | `class InverterCompressor` |
| 5 | ECM fans — airflow law, cube-law power, part-load derate, aero noise | `class ECMFan` |
| 6 | Micro-channel coils — ε-NTU, ṁ^0.78 UA, static head, maldistribution; frost accumulator | `class MicroChannelCoil`, `class FrostModel` |
| 7 | EEV + ejector + IHX + **staged EVI** + liquid quench; valve vs. 180° flip actuator | `class ExpansionTrain`, `class ModeActuator` |
| 8 | Coupled cycle solver — real-gas isentropes, strict energy bookkeeping | `HeatPumpCore.solve`, `HeatPumpCore.step_defrost` |
| 9 | Minimum-noise / minimum-power optimizer — cache, dBA cap, two-stage grid | `class QuietOptimizer` |
| 10 | Lumped room, infiltration, 24-h ambient/solar profiles | `class Room`, `class DayProfile` |
| 11 | Closed-loop PI+feedforward — deadband, night cap, duty cycling, defrost, energy metering | `class ClosedLoopSimulator` |
| 12 | Reporting — rating points, SEER/HSPF bins, quiet-compare, design sweep, spec sheet; **physical spec data** (materials, refrigerant, electrical) | `rating_points`, `seasonal_estimate`, `quiet_compare`, `design_sweep`, `spec_sheet`, `REFRIGERANT_SPEC`, `MATERIALS`, `ELECTRICAL_SPEC`, `PERFORMANCE_SPEC`, `CONTROLS_SPEC`, `SAFETY_SPEC`, `INSTALLATION_SPEC`, `DIMENSIONAL_SPEC`, `ENVIRONMENT_SPEC`, `COMPONENT_DETAIL`, `REGULATORY_SPEC`, `SERVICE_SPEC`, `COST_SPEC` |
| 12b | **Validation harness** — conservation audit, Carnot sweep, property check | `validate`, `R32_PSAT_REF` |
| 12c | **Blueprint export** — frozen JSON design package | `export_blueprint` |
| 12d | **Model drawing** — to-scale geometry, 5-tab pyglet dashboard, OpenGL assembly, SVG export | `DRUM/FRAME/COIL_*`, `loop_segments`, `export_svg`, `run_viewer` |
| 13 | Self-test (15 checks), benchmark, CLI | `self_test`, `_bench`, `main` |

## 4. The Physical Machine — Construction Spec

### 4.1 Geometry (to scale — 0.5 px/mm basis)

| Part | Dimension | Notes |
|---|---|---|
| Window rough opening | 711 × 610 mm class | permanent airtight adapter fills it |
| Adapter frame | 580 × 580 mm section | EPDM perimeter seals, ~+6 W/K total |
| Rotating drum core | **Ø540 mm** | carries *all* internals; horizontal, sill-parallel sealed-bearing axis |
| Coil A (outdoor face) | 52 × 220 mm finned bank | micro-channel, louvered face |
| Coil B (indoor face) | 52 × 220 mm finned bank | same construction |
| Compressor | 120 × 128 mm envelope | twin-rotary inverter, isolation mount |
| Expansion module | 120 × 48 mm | EEV + ejector + IHX + EVI flash tank |
| Outdoor fan | Ø100 mm axial | ECM, oversized/slow |
| Indoor fan | Ø96 mm cross-flow | ECM |
| Condensate tray | 70 × 20 mm | feeds atomizer slinger ring |
| Mass (est.) | ~34 kg installed | drum core ~21 + frame/panels ~13 |

### 4.2 Season reversal — two mechanisms

- **Electronic 4-way valve** (default): `ModeActuator.VALVE_LOSS = 1.2%`
  refrigerant-side penalty, instant, no motion.
- **Physical 180° flip** (`--flip`): the whole drum interior rotates about
  the horizontal axle on sealed bearings. Removes the valve Δp entirely —
  the flip build earns ~1.2% COP. Coil roles swap by *position*: whichever
  face is indoors is the evaporator in summer and the condenser in winter.
  The 2D side-section and 3D assembly both animate the full arc.

### 4.3 Refrigerant — `REFRIGERANT_SPEC`

| Property | Value |
|---|---|
| Name | R-32 (difluoromethane, CH₂F₂) |
| Molar mass | 52.02 g/mol |
| NBP | −51.65 °C |
| Critical point | 78.1 °C / 5.782 MPa |
| GWP₁₀₀ / ODP | 675 / 0 |
| Safety class | A2L (mildly flammable) |
| Charge (est.) | ~450 g — micro-channel, charge-minimized |
| Oil | POE (polyol ester), ~220 ml compressor sump |

**Alternates considered** (`REFRIGERANT_SPEC['alternates_considered']`):
R-290 (GWP≈3 but A3 flammable — charge limit fights a sealed rotating
drum; tracked as future variant) · R-454B (GWP 466, ~10% lower capacity,
near drop-in) · R-410A (GWP 2088, rejected under phasedown) ·
R-744 CO₂ (transcritical, ~10 MPa high side — a different machine entirely).

### 4.4 Bill of construction — `MATERIALS`

| Component | Material / construction |
|---|---|
| Adapter frame | 6063-T5 extruded aluminum, powder-coated, thermal-break polymer insert |
| Drum shell | deep-drawn 5052-H32 aluminum, 1.5 mm wall, closed-cell elastomeric liner at end caps |
| Drum end caps | PA6 + 30% glass fiber, twin louvered apertures |
| Perimeter seals | dual-durometer EPDM rotating lip seals ×2 |
| Bearing axis | 303 stainless axle Ø12 mm; sealed deep-groove bearings (608-2RS class), −40 °C grease |
| Coils | AA3103 aluminum micro-channel flat tube + louvered fin; indoor bank hydrophilic-coated (condensate drainage); both epoxy e-coated |
| Compressor | hermetic steel-shell twin-rotary, NdFeB IPM motor, EVI multi-port shell; elastomeric isolation mount |
| Indoor fan | cross-flow wheel, glass-filled ABS |
| Outdoor fan | axial propeller, glass-filled PP |
| Refrigerant lines | copper ACR 3/8″ + 1/4″, brazed joints; suction line elastomeric-insulated |
| Expansion module | brass-body EEV, machined-brass ejector, brazed-plate 316L stainless IHX, brass EVI flash tank (3 ports) |
| Condensate tray | ABS, sloped to atomizer slinger ring |
| Louvers | extruded aluminum, fixed 20° pitch, both end caps |
| Fasteners | A2-70 stainless |
| Electronics | conformal-coated inverter + controls board, IP54 enclosure at frame lower rail |

### 4.5 Electrical — `ELECTRICAL_SPEC`

| Item | Value |
|---|---|
| Supply | 120/230 V ~ 50/60 Hz universal inverter input |
| Max input | ~1100 W (compressor ~850 + fans ~95 + aux + margin) |
| Standby | 2 W |
| Inrush | negligible — inverter soft-start |
| Cord/plug | NEMA 5-15P (120 V) / per-market cord for 230 V, 1.8 m cord |
| Ingress | electronics bay IP54; drum interior splash-guarded |
| Rated / max current | 6.8 A / 9.2 A @120 V |
| MCA / breaker | 11.5 A minimum circuit ampacity / 15 A branch |
| Power factor | ≥0.98 (active PFC front end) |
| Voltage tolerance | ±10% (108–132 V / 207–253 V) |

### 4.6 Performance spec — `PERFORMANCE_SPEC`

| Item | Value |
|---|---|
| Capacity class | 12 kBTU/h (nominal 3.5 kW cool / 3.4 kW heat) |
| Rated cooling | 3539 W @ 35 °C amb / 26.7 °C room — COP 4.86 |
| Rated heating | 3432 W @ 8.3 °C amb / 21.1 °C room — COP 4.61 |
| Modulation | 1.7–5.0 kW continuous (15–120 rps inverter); duty-cycled below ~1.7 kW |
| Airflow indoor | 0.55 kg/s ≈ 980 CFM nominal @ 750 rpm (oversized cross-flow) |
| Airflow outdoor | 0.96 kg/s ≈ 1720 CFM nominal @ 600 rpm (axial) |
| Dehumidification | ≈2.2 L/h latent @ 26.7 °C/50% RH room, 35 °C amb (model-derived) |
| Ambient envelope | cool +10…+46 °C; heat −50…+24 °C (EVI×3 + quench, T_dis ≤112 °C) |
| Sound | indoor 17–36 dBA; 25 dBA night cap; outdoor <50 dBA |
| Seasonal | SEER ≈22.4 · HSPF ≈12.6 (bin estimate) |

### 4.7 Controls — `CONTROLS_SPEC`

- **Board:** conformal-coated inverter + MCU in IP54 bay.
- **Sensors:** indoor/outdoor dry-bulb, coil-face thermistors,
  suction/discharge pressure transducers, line current, fan tach;
  frost proxy via coil-T + ambient-RH.
- **Algorithms:** PI + feedforward capacity control, ±0.15 K deadband;
  minimum-noise operating-point optimizer; duty cycling below 1.6×
  overshoot (8% restart penalty); adaptive reverse-cycle defrost
  (metered, ≤4 min); EVI staging + discharge-temperature quench control.
- **Interfaces:** IR remote + capacitive panel on indoor face; Wi-Fi
  footprint for app/demand-response (OpenADR-ready).
- **Modes:** cool / heat / auto / fan-only / night-quiet (25 dBA cap
  22:00–07:00) / dry-bias.

### 4.8 Safety — `SAFETY_SPEC`

- **Flip interlock:** rotation inhibited while powered; dual locking
  pins + positional microswitch confirm 0°/180° detent before restart.
- **Seal monitor:** perimeter-seal compression switch, fault if
  unseated >3 mm.
- **Refrigerant safety:** 450 g A2L charge below the flammability limit
  for the smallest install room; leak-detect sensor in indoor bay with
  auto-shutoff + alarm.
- **Pressure:** high-pressure cutout 4.2 MPa; ISO 5149-sized relief on
  the liquid line.
- **Thermal:** 120 °C discharge limit (quench-enforced) + motor
  thermal protector.
- **Mechanical:** anti-tip rails, <8 mm louver-guard spacing,
  finger-safe fan shrouds.
- **Electrical:** GFCI-compatible cord, fused board, surge clamp.

### 4.9 Installation & service — `INSTALLATION_SPEC`

- **Opening fit:** double-hung / slider / casement, 711×610 mm class via
  telescoping adapter frame; custom panels for full sash replacement.
- **Weight:** ~34 kg installed; gas-strut-assisted pivot → one-person
  seasonal flip.
- **Daylight:** optional translucent polycarbonate side panels preserve
  ~60% of glazed area around the core.
- **Sealing:** EPDM perimeter + thermal break → 6 W/K infiltration
  (vs 30–50 typical).
- **Service:** indoor-only washable MERV-8 electrostatic filter;
  factory-sealed refrigerant circuit — no field charge.
- **Durability:** 15–20 yr design life; e-coated coils, stainless
  fasteners, UV-stabilized polymers.

### 4.10 Dimensions — `DIMENSIONAL_SPEC`

| Item | Value |
|---|---|
| Installed envelope | frame face 711 W × 610 H mm fills rough opening; drum OD 540 mm |
| Depths | indoor projection 120 mm; outdoor 260 mm; ~380 mm through wall + sill bracket |
| Clearances | indoor 300 mm front (filter swing + service); outdoor 600 mm discharge |
| Sill slope | ±2° tolerance; shims compensate to 4° out-of-level |
| Opening types | double-hung / slider / casement via telescoping adapter; full sash-replacement optional |
| Shipping | 780×700×480 mm carton, ~40 kg gross; pre-charged core |

### 4.11 Environmental limits — `ENVIRONMENT_SPEC`

| Item | Value |
|---|---|
| Operating | cool +10…+46 °C; heat −50…+24 °C; RH ≤90% non-condensing indoor |
| Storage | −30…+60 °C, RH ≤95% non-condensing |
| Altitude | ≤2000 m full rating; ~1%/300 m derate above (density-corrected in model) |
| Weather | driven rain, −50 °C cold-soak start, 40 m/s gust |
| Corrosion | 1000 h salt-spray coils (e-coat); coastal package optional |

### 4.12 Regulatory targets — `REGULATORY_SPEC`

*Design targets — certification path identified, nothing certified yet.*

- **Safety:** UL 60335-2-40 Ed.5 (A2L clauses) / IEC 60335-2-40.
- **Refrigerant:** ISO 5149 / EN 378 charge limits — 450 g A2L within
  the unventilated-room limit for >4 m² floor area.
- **Performance:** AHRI 210/240 methods; DOE CEER procedure (10 CFR 430
  subpart B).
- **Emissions:** FCC Part 15 Class B; CISPR 14-1.
- **Chemistry:** RoHS 3 / REACH; no PFAS coatings; SNAP-listed R-32.
- **Label:** ENERGY STAR room-AC criteria exceeded ~1.8× (CEER 15 req →
  ~26 seasonal-equivalent modeled).

### 4.13 Service & warranty — `SERVICE_SPEC`

| Item | Value |
|---|---|
| Warranty (target) | 2 yr parts / 5 yr compressor |
| Maintenance | filter rinse monthly in season; annual seal check; self-cleaning condensate path |
| Service ports | factory-sealed circuit; two Schrader stubs behind cover — recovery only, no field charge |
| Diagnostics | onboard LED fault codes + app history; sensor-fail → safe fixed-speed fallback |
| Reliability | design life 15–20 yr; bearings/seals >10 yr; flip mechanism >50 cycles (~25 yr seasonal margin) |

### 4.14 Component detail — `COMPONENT_DETAIL`

| Item | Value |
|---|---|
| Coil geometry | 220 mm face × 52 mm deep × 3 tube rows; 20 FPI louvered 0.10 mm fins; multiport flat tubes 1.0×16 mm, 24 ports/tube |
| Design pressures | high side 4.5 MPa working / 6.75 proof; low side 2.5 / 3.75 (ISO 14903 class A) |
| Leak rate | <1 g/yr target — all-brazed joints, no flares in field path |
| EEV | 500-step pulse valve, 0–2.4 mm travel, closed-loop on 5 K superheat + 6 K subcool |
| Compressor shell | Ø118 × 170 mm steel shell, 12 cm³ twin-rotary, POE 220 ml, EVI×3 + quench |
| Fan motors | ECM BLDC 24 V bus, class-F insulation, ball bearings, IP42, tach feedback |
| Bearings & axis | 608-2RS, L10 >40 000 h; Ø12 mm 303 axle; ~180 N·m flip torque; 2×350 N gas struts |
| Insulation | 6 mm closed-cell NBR/PVC (λ≈0.034 W/mK) in end caps + frame thermal break |
| Filter | washable MERV-8 electrostatic 400×180 mm, indoor access only |
| Finish | powder-coat RAL 7016 graphite 60 µm; anodized drum; matte indoor face |

### 4.15 Cost targets — `COST_SPEC`

| Item | Value |
|---|---|
| BOM estimate | ~$520 (modeled target) |
| MSRP target | $1299 — premium position vs $350 commodity AC |
| Install | DIY ~$0 / pro ~$150 (15-min adapter frame) |
| Annual energy | ≈$95 cooling + $140 heating @ $0.16/kWh mixed climate (bin est.) |
| Payback | ~3 yr vs window AC + resistance heat |

## 5. Module-Level Constants (§1)

| Constant | Value | Use |
|---|---|---|
| `R_AIR` | 287.05 J/(kg·K) | air density from ideal gas |
| `CP_AIR` | 1006.0 J/(kg·K) | air-side capacity rates |
| `RHO_AIR` | 1.18 kg/m³ | fan-flow bookkeeping |
| `P_ATM` | 101325 Pa | psychrometrics |
| `HFG_WATER` | 2.501e6 J/kg | condensate latent split |

Helpers: `K(c)` °C→K · `C(k)` K→°C · `F(k)` K→°F ·
`db_sum(*levels)` incoherent power sum `10·log₁₀(Σ10^(L/10))`, −∞-safe.

## 6. Data Structures

### `OperatingPoint` — a commanded hardware state

| Field | Unit | Meaning |
|---|---|---|
| `n_rps` | rev/s | compressor speed (15–120) |
| `rpm_in` | rpm | indoor ECM fan speed |
| `rpm_out` | rpm | outdoor ECM fan speed |

### `CycleResult` — everything the solver knows about one steady point

| Field | Unit | Meaning |
|---|---|---|
| `ok` | — | solve converged and feasible |
| `mode` | — | `'cool'` or `'heat'` |
| `n_rps` | rev/s | echo of command |
| `t_evap` / `t_cond` | K | converged saturation temperatures |
| `t_discharge` | K | compressor discharge (real-gas) |
| `p_low` / `p_high` | Pa | coil saturation pressures |
| `m_dot_ref` | kg/s | refrigerant mass flow (compressor-side) |
| `q_capacity` | W | delivered to conditioned side |
| `q_sensible` | W | dry-bulb capacity (cooling: total − latent − indoor fan heat) |
| `q_latent` | W | moisture-removal capacity |
| `q_reject` | W | rejected to the other side |
| `w_compressor` | W | electrical at compressor terminals |
| `w_thermo` | W | refrigerant-side work (audit quantity) |
| `q_shell_w` | W | shell heat lost to ambient (audit term) |
| `w_fans` / `w_fan_in` | W | total fan draw / indoor-fan share dissipated to room air |
| `w_aux` | W | standby + atomizer + crankcase |
| `w_total` | W | all electrical input |
| `cop` | — | q_capacity / w_total |
| `carnot_cop` / `carnot_fraction` | — | reference bound, and fraction of it |
| `air_delta_t` | K | conditioned-coil air approach |
| `m_air_in` / `m_air_out` | kg/s | density-corrected coil airflows |
| `condensate_kg_s` | kg/s | cooling only |
| `dba_indoor` / `dba_outdoor` | dBA | A-weighted @1 m |
| `evi_active` / `defrosting` / `frost` | — | state flags |
| `actuation` | — | `'valve'` or `'flip'` |
| `notes` | str | human-readable flags (`[EVI]`, `[quench y=…]`) |

## 7. R-32 Property Model (`RefrigerantR32`, §2)

Correlation-grade single-component model — *not* REFPROP, but real-gas
on the vapor path. Accuracy ±3–6% vs REFPROP across −40…+65 °C.

### 7.1 Constants

| Symbol | Value |
|---|---|
| `T_CRIT` | 351.255 K |
| `P_CRIT` | 5.782 MPa |
| `M` | 0.05202 kg/mol |
| `R` | 8.314462 / M ≈ 159.83 J/(kg·K) |
| `OMEGA` (ω) | 0.2269 (acentric) |
| `T_REF` datum | 233.15 K, h_l = 200 kJ/kg, s = 1.0 kJ/(kg·K) |

### 7.2 Saturation correlations

- **`psat(t)`** — Wagner equation anchored at the critical point:
  `ln(p/Pc) = (Tc/t)·Σ aᵢ·τⁿⁱ` with τ = 1−t/Tc, coefficients
  `a = (−6.4325, −1.2272, 8.985)`, exponents `n = (1, 1.5, 4.5)`.
  Re-anchored to NBP −51.65 °C @ 0.1013 MPa + 0 °C/25 °C reference points;
  worst error vs. table **3.0%** (old coefficients were 11–17% off at the
  cold end — caught by `--validate`).
- **`tsat(p)`** — Newton inversion of psat, 60-iteration cap, clamped
  [180 K, Tc−0.01].
- **`hfg(t)`** — Watson correlation anchored at 253.15 K, 352.0 kJ/kg,
  exponent 0.38.
- **`cp_liq(t)`** — `1550 + 7·(t−253.15)` J/(kg·K), clamped [1200, 3000].
- **`cp_gas(t)`** — `500 + 1.2·t` J/(kg·K).
- **`h_liq`, `h_g_sat`, `s_liq`, `s_g_sat`** — datum-consistent
  (s_g = s_l + hfg/T, so isentropic compression lands correctly).
- **`h_g_super`, `s_g_super`** — superheat offsets from saturation base.

### 7.3 Peng–Robinson vapor path (the real-gas pass)

- **`_pr_coefs(t)`** — α-correlation `κ = 0.37464 + 1.54226ω − 0.26992ω²`;
  `a_c = 0.45724·R²Tc²/Pc`; `a = a_c·m²` with `m = 1 + κ(1−√Tr)`;
  `b = 0.07780·RTc/Pc`; returns `(a, b, da/dT)`.
- **`z_vapor(t, p)`** — largest root of the PR cubic, **analytic Cardano
  solve** (three-real-root trigonometric branch + one-real-root cubic
  branch) plus one Newton polish step; `z > B` guard prevents falling
  into the liquid root. ~5× faster than the old 40-iteration Newton.
- **`_dep(t, p)`** — enthalpy/entropy **departure functions**:
  `h_dep = RT(z−1) + (T·da/dT − a)/(2√2·b)·ln[(z+(1+√2)B)/(z+(1−√2)B)]`,
  `s_dep = R·ln(z−B) + (da/dT)/(2√2·b)·ln[…]`.
- **`h_ideal(t)`** — `350·(t−273.15) + 0.8·(t²−273.15²)` (cp0 = 350+1.6T).
- **`s_ideal(t, p)`** — `350·ln(t/T0) + 1.6·(t−T0) − R·ln(p/P0)`.
- **`h_vap(t, p)`** — `h_ideal + h_dep + H_ANCHOR`, anchored once onto the
  `h_g_sat` datum at 280 K / psat(280 K) so both scales agree (the
  datum-mismatch fix that closed the EVI conservation leak).
- **`s_vap(t, p)`** — `s_ideal + s_dep + S_ANCHOR` (same anchoring).
- **`_solve_t_for_s` / `_solve_t_for_h`** — Newton inversions
  (40-iteration cap, clamp [200, 600] K) used for isentropic discharge
  and stage-mixing inside EVI.

## 8. Psychrometrics (§3)

- **`p_ws(°C)`** — Magnus formula `610.94·exp(17.625·t/(t+243.04))` Pa.
- **`w_ratio(°C, rh)`** — humidity ratio `0.622·p_w/(P_atm−p_w)`.
- **`condensate_rate(m_air, t_in, rh_in, t_evap)`** — coil dew surface
  ≈ t_evap+2 K at 95% RH outlet; `ṁ_cond = m_air·(w_in − w_out)⁺`;
  returns (kg/s, latent W = ṁ_cond·HFG_WATER).

## 9. Compressor (`InverterCompressor`, §4)

Twin-rotary, vapor-injection ready. Quiet-design proxies: wide-inlet
low-velocity ports, magnetic-bearing proxy (no rubbing floor below
~35 rps), discharge muffler.

| Parameter | Value | Role |
|---|---|---|
| `V_D` | 12 cm³ | swept volume |
| `N_MIN` / `N_MAX` | 15 / 120 rps | inverter range |
| `N_PEAK` | 60 rps | apex of η_is parabola |
| `CLEARANCE` | 0.05 | re-expansion fraction |
| `K_GAS` | 1.15 | polytropic proxy in η_vol |
| `ETA_IS_PEAK` | 0.80 | isentropic ceiling |
| `ETA_EL` | 0.93 | motor+inverter at rated speed |

Modeled behavior:

- `ṁ = ρ·V_D·N·η_vol` with **real-gas** `ρ = p/(Z·R·T_suc)` —
  suction density from `z_vapor`, not ideal.
- `η_vol = max(1 − 0.05·(pr^(1/1.15) − 1), 0.55)`.
- `η_is = 0.80·f_speed·f_pr`;
  `f_speed = clamp(1 − 0.28·((N−60)/60)², 0.55, 1)`;
  `f_pr = clamp(1 − 0.05·(pr−3), 0.60, 1)`.
- `η_el(N) = 0.93·(0.82 + 0.18·min(N/60, 1))` — sags ~18% at minimum
  speed (stator/drive losses dominate deep part-load).
- Noise @1 m: `30 + 24·log₁₀(N/25) + 3·log₁₀(pr/3)` dBA, floor 24 dBA.
- Electrical input: `W_el = ṁ·w_act·(1 − ejector_recovery)/η_el` —
  the ejector contributes real mechanical energy before the motor does.

## 10. Fans (`ECMFan`, §5)

Oversized + slow = the quiet lever; rated airflow arrives at low rpm.

| | Indoor (cross-flow) | Outdoor (axial) |
|---|---|---|
| `k` (kg/s per rpm) | 7.3e-4 | 1.6e-3 |
| `rpm_nom` | 750 | 600 |
| `p_nom` | 38 W installed draw | 55 W installed draw |
| `l_nom` | 34 dBA @1 m | 46 dBA @1 m |
| `eta` | 0.62 | 0.62 |

- `airflow = k·rpm` (linear fan law).
- `power = p_nom·(rpm/rpm_nom)³/η_eff` — cube law; `p_nom` is the
  *installed* operating-point draw, so coil static head is already inside
  the law (the dp·V̇ double-count was found and removed).
- Part-load: `η_eff = 0.62·(0.85 + 0.15·min(rpm/rpm_nom, 1))` — ECM loses
  ~15% below ~40% nominal speed.
- Noise: `l_nom + 55·log₁₀(rpm/rpm_nom)`; 10 dBA below 1 rpm.

## 11. Coils (`MicroChannelCoil`, `FrostModel`, §6)

| | Indoor `coil_in` | Outdoor `coil_out` |
|---|---|---|
| `ua_nom` | 900 W/K @ 0.55 kg/s | 1400 W/K @ 0.90 kg/s |
| `dp_ref` | 60 Pa @ m_air_ref | 60 Pa @ m_air_ref |

- `UA = ua_nom·(ṁ/ṁ_ref)^0.78·MALDIS`, `MALDIS = 0.94` — the louvered
  drum face never sees perfectly even flow; honest 6% derate.
- `dp_static = dp_ref·(ṁ/ṁ_ref)²` — turbulent-regime coil property.
- `ε = 1 − exp(−NTU)`, `NTU = UA·(frost·wet·wind)/ (ṁ·cp_air)`;
  phase-change side isothermal at T_sat.
- **Wet coil:** indoor coil below room dew point → latent release at the
  fins boosts effective UA ×1.10.
- **`FrostModel`** — 0..1 accumulator; growth
  `0.012·(rh−0.55)/0.45·(1−|t+2|/10)`/min inside −8<t<4 °C & RH>55%;
  passive melt 0.002/min; defrost clears ~1 min; `ua_derate = 1−0.45·lvl`.

## 12. Expansion Train (`ExpansionTrain`, `ModeActuator`, §7)

| Element | Parameter | Value |
|---|---|---|
| EEV | `SUBCOOL_K` | 6 K |
| Suction | `SUPERHEAT_K` | 5 K |
| IHX | `IHX_EFF` | 0.45 (energy-conserving: liquid pre-cool = suction pre-heat) |
| Ejector | `EJECTOR_WORK_RECOVERY` | 0.06 (credited against electrical input) |
| EVI | `EVI_INJECTION` | 0.13 base, **lift-scaled** `x = 0.13·max(1, pr/5)`, cap 0.32/port |
| 4-way valve | `VALVE_LOSS` | 0.012 (0 in flip mode) |

## 13. Cycle Solver (`HeatPumpCore.solve`, §8)

### 13.1 Loss/environment fields

| Field | Value |
|---|---|
| `dp_suc` / `dp_dis` | 15 kPa / 30 kPa at `m_dot_ref` = 10 g/s — **scale ṁ²** each solve |
| `shell_frac` | 0.04 of refrigerant-side work → ambient |
| `w_crankcase` | 25 W below −5 °C (running *or* idle) |
| `w_standby` / `w_atomizer` | 2 W / 3 W |
| `wind_ua_gain` | 1.05× outdoor-coil UA (free forced convection) |
| `t_dis_max` | 393 K (120 °C); quench target `t_lim` = 385 K (112 °C) |
| `evi_threshold_c` | −3 °C |

### 13.2 Iteration structure (damped fixed-point, ≤80 passes)

1. Assign coil roles by mode; density-correct both airflows
   (`ρ = P/(R·T)`, normalized to 1.2 kg/m³ — at −50 °C the outdoor coil
   sees ~35% more kg/s per rpm).
2. Winter solar assist: `solar_w/(ṁ_evap·cp)` added to evaporator air T.
3. EVI staging: `n_ports = 1` (<−3 °C), `2` (<−18 °C), `3` (<−30 °C).
4. Warm-start `(t_e, t_c)` from `self._guess[mode]` (else air ±9/11 K).
5. Per iteration: compressor-side pressures `p1c = p_sat(t_e) − dp_suc·f_s`,
   `p2c = p_sat(t_c) + dp_dis·f_s` with `f_s = (ṁ_lag/ṁ_ref)²` — line losses
   lagged one step.
6. Evaporator exit at +5 K superheat (real `h_vap`); IHX suction-side gain
   `ihx_gain`; compression path —
   - single-stage: isentrope via `_solve_t_for_s`, work via `h_vap` deltas;
   - EVI: march `n_ports` stages — compress to geometric-split port
     pressure `p_i = p1c·pr^((i+1)/(n+1))`, mix in `x_i` kg of saturated
     flash vapor (exact per-port mass/enthalpy balance on the PR datum),
     then final stage to discharge.
7. **Liquid quench** (EVI only): if `t2 > t_lim`, bisection on quench
   fraction `y ≤ 1.2` kg/kg-suction — subcooled liquid drawn off the
   liquid line flashes at the last port and clamps `t2 ≤ 112 °C`.
   Warm-started bracket from the previous iteration's `y`. Two-phase
   mixes clamp entropy at saturated vapor (conservative under-credit).
8. Condenser energy `q_cond = ṁ_out·(h2 − h3_pre)`; shell loss
   `q_shell = shell_frac·W_thermo` deducted on the ambient side.
9. Condensate assist (cooling): `ΔT_assist = q_evap_avail/(ṁ_cond·cp)·0.65`,
   capped 6 K — condenser airstream pre-cooled toward wet-bulb.
10. Coil balances: `Q_evap = ε_e·ṁ_e·cp·(air − t_e)`,
    `Q_cond = ε_c·ṁ_c·cp·(t_c − air)`; damped update of `(t_e, t_c)`;
    converged → populate `CycleResult`.
11. Noise: `dba_in = db_sum(comp − 10 dB, fan_in)`,
    `dba_out = db_sum(comp, fan_out)`; aux `w_aux` assembled
    (standby + atomizer + crankcase + valve).

### 13.3 Conservation

The audit identity `Q_evap + W_thermo = Q_cond + Q_shell` closes to
**0.000%** worst case across five audit points including the −45 °C
quench path. Two datum bugs were found and fixed to get there: vapor
injection credited on the correlation scale vs. PR scale, and the
quench draw omitting its liquid-line enthalpy share.

## 14. Optimizer (`QuietOptimizer`, §9)

- Grids: `N ∈ 15–120 rps` (step 5) × `rpm_in ∈ 200–900` (step 50) ×
  `rpm_out ∈ 250–900` (step 50).
- **Two-stage scan:** coarse grid, then refine ±5 rps / ±50 rpm around
  the winner — verified to land the identical optimum as exhaustive
  search; `opt.best` ≈ 0.79 s, ~5,700 solves/s.
- `best()` — minimum feasible `dba_indoor` (night `dba_cap` filters);
  `best_power()` — minimum `w_total`; feasibility uses room-side
  `q_sensible`/`q_capacity` net of fan-motor heat.
- Result cache keyed on (mode, rounded temps, load, cap).

## 15. Room & Profiles (§10)

- **`Room`**: `ua = 150 + 6 = 156 W/K`, `c_eff = 4.2 MJ/K`,
  `step(dt, q_hvac, t_out, q_internal=250 W, q_solar)` —
  `ΔT = (q_hvac + q_int + q_sol − ua·ΔT_env)·dt/c_eff`.
  `steady_load(t_set, t_out, ua, q_int, q_sol)` analytic load.
- **`DayProfile(t_min, t_max, peak_hour=15, rh)`** — sinusoidal ambient
  peaking 15:00; solar `400·sin(π·(h−7)/12)` W for 7<h<19.

## 16. Closed-Loop Controller (`ClosedLoopSimulator`, §11)

120 s steps over 24 h. Per step:

- `err = t_room − t_set`; inside ±0.15 K deadband → coast (fans off,
  standby + crankcase only).
- Request `q_req = max(0, ff + 900·err + 6·∫err)` — PI + feedforward on
  the analytic steady load.
- **Night cap**: 22:00–07:00 with `--night`, `dba_cap = 25`.
- **Duty cycling**: if the quietest feasible point delivers
  `q_sensible > 1.6×q_req` → run fraction `duty = q_req/q_sensible`,
  energy `×1.08` restart penalty, Leq noise `+10·log₁₀(duty)` —
  short quiet bursts beat a continuous drone.
- **Defrost**: `step_defrost` accrues frost; level > 0.55 → reversed-cycle
  defrost ≤240 s or until ≤0.02; energy metered separately.
- Energy buckets: `hvac_kwh`, `comp_kwh`, `defrost_kwh`,
  `resist_kwh` (COP-1 baseline for comparison).

## 17. Operating Envelope (verified this build — `--rate`)

| Ambient | Delivered | Input | COP | Carnot% | Noise | Notes |
|---|---|---|---|---|---|---|
| +35 °C cool / 80 °F room | 3539 W | 728 W | 4.86 | 46% | 27.4 dBA | full load |
| +28 °C cool / 80 °F room | 2071 W | 285 W | 7.26 | 42% | 18.1 dBA | part load |
| +8 °C heat / 70 °F room | 3432 W | 744 W | 4.61 | 52% | 26.2 dBA | |
| −8 °C heat | 3427 W | 1014 W | 3.38 | 51% | 30.7 dBA | EVI, quench y=0.04 |
| −15 °C heat | 2245 W | 711 W | 3.16 | 54% | 29.1 dBA | EVI, quench y=0.10 |
| −40 °C heat | 1502 W | 777 W | 1.93 | 44% | 30.2 dBA | EVI×3, quench y=0.55 |
| **−50 °C heat** | 1545 W | 849 W | **1.82** | 48% | 35.8 dBA | EVI×3, quench y=0.93, T_dis ≤112 °C |

Seasonal bins — cooling `(19,22,25,28,31,34,37 °C)` weights
`(0.05,0.15,0.25,0.25,0.18,0.09,0.03)`; heating
`(−12,−7,−2,3,8,13,16 °C)` weights `(0.03,0.10,0.20,0.25,0.22,0.15,0.05)`:
cooling COP ≈ 6.56 → **SEER ≈ 22.4**; heating COP ≈ 3.69 → **HSPF ≈ 12.6**.

Closed-loop days: hot day (25–38 °C) ~11.5 kWh at avg COP ~5.2, ~28 dBA;
cold day (−6…+3 °C) ~20 kWh at avg COP ~4.0 with metered defrost; mild
day duty-cycles (1.7 kWh, avg COP 6.7, 9.3 dBA Leq).

## 18. Validation Harness (`validate`, §12b)

Three independent proofs — `python AC.py --validate`:

1. **Conservation audit** at 5 points: cool +35/+45 °C, heat +8/−15/−45 °C.
   `|Q_evap + W_thermo − Q_cond − Q_shell| / Q_capacity < 1%`.
   Current worst: **0.000%**.
2. **Carnot-bound sweep**: 2 modes × 6 ambients (−20,−5,10,25,40,46 °C)
   × 3 speeds (30,70,115 rps) = 36 points; any `cop ≥ carnot_cop` is a
   violation. Current: **0 violations**.
3. **R-32 psat vs. `R32_PSAT_REF`**: 6 reference points
   (−51.65, −40, −20, 0, +25, +40 °C); each must be within 8%.
   Current worst: **3.0%**.

`VERDICT: PASS` requires all three.

## 19. Self-Test (`self_test`, §13) — all 15 checks

1. psat sanity: ~0.79 MPa @ −3 °C, ~1.7 MPa @ +27 °C
2. `tsat` inversion within 0.5 K
3. `hfg` monotonic decrease toward Tc
4. entropy ordering `s_g > s_l`
5. cycle solves in both modes, COPs in plausible bands; −15 °C → EVI on
6. flip build beats valve build (no valve Δp)
7. optimizer finds a feasible point < 40 dBA
8. `db_sum(30,30) = 33.01 dBA`
9. condensate rate non-negative
10. frost accrues in the frosting window and derates UA
11. IHX within ±8% COP, never crashes
12. EVI genuinely adds capacity at −15 °C
13. latent split bounded `0 ≤ latent ≤ capacity`
14. night cap 25 dBA honored when feasible
15. **arctic**: `opt.best("heat", 21 °C room, −50 °C, 1400 W)` returns
    `ok`, COP > 1, capacity ≥ 1400 W

## 20. Reporting Modes (§12)

- **`--rate`** — `rating_points`: 7 rows (§16 table) via `opt.best`.
- **`--season`** — `seasonal_estimate`: SEER2/HSPF2-like bin method.
- **`--quiet-compare`** — same 3.2 kW load, loud-fast point
  (N-scanned at 850/850 rpm fans) vs. optimized-quiet → ΔdBA + Δpower.
- **`--optimize`** — `design_sweep`: coil UA ×{0.80,1.00,1.25} × fan
  ×{0.85,1.00,1.15} × Vd {10,12,14 cm³} = 27 feasible candidates;
  score = seasonal-weighted W + 0.3·dBA_max·10 + packaging penalty
  (`60·(ua−1) + 20·(fan−1)`, hard cap ua ≤1.30 — the drum physically
  can't fit more). Rating cases: cool95/cool82/heat47/heat17 weighted
  0.30/0.30/0.22/0.18. Current answer: UA×1.25, fan×1.15, Vd=10 cm³.
- **`--spec`** — `spec_sheet`: one-page spec — hardware, refrigerant
  chemistry + charge, electrical ratings, full materials bill of
  construction.
- **`--bench`** — grid-solve throughput.

## 21. Visualization (§12d)

### 21.1 To-scale geometry (0.5 px/mm, canvas 1440×900)

| Object | Model dims |
|---|---|
| Wall/opening band | ~200 mm thick |
| Adapter frame | 580×580 mm section |
| Drum | **Ø540 mm** (`DRUM r=270 mm`) |
| Coil A / Coil B | 52×220 mm each |
| Compressor | 120×128 mm |
| Expansion module | 120×48 mm |
| Fan A (outdoor) | Ø100 mm axial |
| Fan B (indoor) | Ø96 mm cross-flow |
| Condensate tray | 70×20 mm |

Ink palette: `BG_GREY (62,63,68)`, `INK (235,236,240)`,
`INK_DIM (168,170,176)`, `ENVELOPE (90,92,98)`, `HOT`, `COLD`.
`FLOW_COLORS` maps 7 refrigerant states (discharge→suction).

### 21.2 Tabbed dashboard (`run_viewer`)

- Window: 1440×900 **resizable**, launches **fullscreen** (`F11`
  toggles; `ESC` drops to windowed first, then quits). All 2D content is
  authored in a fixed 1440×900 model space and **letterboxed** to any
  window size via a `win.view` scale+translate matrix — mouse hits and
  projected 3D labels take the inverse transform, so tabs and part
  labels stay accurate at any size.
- GL context: `Config(major_version=2, minor_version=1,
  double_buffer=True, depth_size=24)` — the NVIDIA driver honors this as
  a 4.6-compatibility context so pyglet shaders (2D) and fixed-function
  GL (3D) coexist. `double_buffer` is explicit: a single-buffered format
  makes `glDrawBuffer(GL_BACK)` fail and immediate-mode geometry is
  never presented.
- Batches: `chrome` (title + tab bar), `bg_b` (fullscreen grey, drawn
  *before* the 3D pass), `static`/`batch` (MODEL), `cyc_b` (CYCLE),
  `spe_b` (SPECS), `val_b` (VALIDATION), `leg_b` (3D legend/labels).
- **Pyglet 2.1.x quirk worked around**: GL buffers created before the
  context is current silently drop the initial vertex upload — every
  created shape is collected and `.x = .x` self-assigned once inside the
  event loop to force re-upload (`primed` list).
- Tabs (`1–5`, `TAB`, or click): **MODEL** (side section, live
  refrigerant particles by state, spinning fans, flip animation,
  telemetry column: T_evap/T_cond/T_dis/p/ṁ/Q/W/COP/Carnot%/dBA/EVI),
  **CYCLE** (live log-p/h dome + 4-point polygon + state table),
  **SPECS**, **VALIDATION** (harness output rendered in-window),
  **3D** (below).
- Keys on MODEL: `←/→` compressor ±5 rps · `↑/↓` fans · `c`/`h` mode ·
  `f` flip/valve actuation · `o` quiet-optimize · `space` pause ·
  `F11` fullscreen · `esc` windowed→quit.

### 21.3 3D assembly (tab 5)

PyOpenGL fixed-function pipeline: `gluPerspective`, `GL_LIGHTING` +
`GL_LIGHT0/1`, `GL_DEPTH_TEST`, `glPushMatrix`/`glTranslatef` per part.
Parts: wall slab, adapter frame, translucent drum + louvered end caps +
rim rings + axle + sealed bearings, both coil banks with fins, axial +
cross-flow fans (spin at real rpm), compressor + mount ring, expansion
train, condensate tray, copper piping runs. Drag to orbit
(`rx/ry` state), wheel to zoom (`zoom`), `x` toggles the animated
exploded view; part labels project through `gluProject` and track their
components. `f` + `h` still performs the 180° drum flip in `--flip` mode.

### 21.4 SVG export (`export_svg`, `--svg`)

Three-panel sheet: **Panel A** side section (summer) with wall, frame,
drum internals, refrigerant loop, airflow arrows, 180° flip arc,
dimension lines, scale bar; **Panel B** front elevation (711×610 mm
opening, louvered drum face, bearing caps); **Panel C** summer/winter
flip states. Refrigerant-state legend + live telemetry snapshot.

## 22. Blueprint Export (`export_blueprint`, §12c, `--blueprint`)

Writes `AURA-1_blueprint.json` — the frozen as-designed record. Schema:

| Top-level key | Contents |
|---|---|
| `unit` | product string |
| `actuation` | `'valve'` or `'flip'` |
| `refrigerant` | full `REFRIGERANT_SPEC` dict + `property_model` (PR EOS, Wagner, Watson) |
| `materials` | full `MATERIALS` bill of construction (15 components) |
| `electrical` | full `ELECTRICAL_SPEC` (supply, max/rated current, MCA, breaker, power factor, cord, IP rating, mass) |
| `performance_spec` | capacity class, rated points, modulation, CFM, dehumidification, ambient envelope, sound, seasonal |
| `controls_spec` | board, sensor suite, algorithms, interfaces, modes |
| `safety_spec` | flip interlock, seal monitor, A2L leak detect, pressure/thermal/mechanical/electrical protection |
| `installation_spec` | opening fit, weight/flip assist, daylight panels, sealing, service access, durability |
| `dimensional_spec` | envelope, projections, clearances, sill tolerance, shipping |
| `environmental_spec` | operating/storage limits, altitude, weather, corrosion |
| `regulatory_spec` | certification targets: UL/IEC 60335-2-40, ISO 5149, AHRI/DOE, FCC/CISPR, RoHS/REACH, ENERGY STAR |
| `service_spec` | warranty targets, maintenance, service ports, diagnostics, reliability ratings |
| `component_detail` | coil geometry, design/proof pressures, leak rate, EEV, shells, motors, bearings, insulation, filter, finish |
| `cost_spec` | BOM, MSRP, install, annual energy $, payback |
| `geometry` | drum_od 540 mm, frame 580×580, opening 711×610, flip axis + EPDM |
| `compressor` | type, `v_d_cm3` 12, `n_range_rps` [15,120], `eta_is_peak`, `t_dis_max_c` |
| `coils` | indoor/outdoor `ua_nom_w_k`, `m_air_ref`, `dp_ref_pa`, `maldis_derate` |
| `fans` | indoor/outdoor `kg_per_rpm`, `rpm_nom`, `p_nom_w`, `l_nom_dba` |
| `expansion` | subcool/superheat/IHX/ejector, EVI base + staging thresholds, quench description |
| `losses` | dp_suc/dp_dis at ref, ṁ² scaling, shell 4%, crankcase 25 W, valve 1.2%, standby/atomizer |
| `aux` | wind_ua_gain 1.05, solar-coil model, evi_threshold −3 °C |
| `operating_envelope` | 6 verified snapshots: `cool_35C`, `heat_8C`, `heat_-8C`, `heat_-15C`, `heat_-40C`, `heat_-50C` — each `q_w`, `w_w`, `cop`, `dba_in`, `t_dis_c`, `ok`, `evi`, `notes` |
| `control_law` | optimizer description, duty-cycle rule, defrost rule, night cap 25 dBA |

## 23. CLI (`main`, §13)

| Flag | Action |
|---|---|
| *(none)* | banner + launch viewer |
| `--test` | 15-check self-test |
| `--bench` | optimizer throughput |
| `--rate` | rating table |
| `--season` | SEER/HSPF bins |
| `--quiet-compare` | loud vs. quiet strategy |
| `--optimize` | hardware design sweep |
| `--blueprint` | write JSON design package |
| `--spec` | spec sheet |
| `--validate` | conservation + Carnot + property proofs |
| `--day cool\|heat` | 24-h closed-loop sim (hot/cold profile) |
| `--night` | +25 dBA cap 22:00–07:00 (with `--day`) |
| `--flip` | physical 180° actuator instead of valve |
| `--svg` | write `AURA-1_diagram.svg` |
| `--view` | launch viewer explicitly |
| `--all` | all console modes, then viewer |

Console re-encoded to UTF-8 on start (Windows cp1252 would mangle the
symbols in reports).

## 24. Honesty Layer

- **Real-gas accurate** on the vapor path (PR EOS), but not REFPROP:
  saturated-liquid side uses anchored correlations; two-phase inside
  coils is isothermal-lumped; cp0 is a 2-constant fit. Absolute values
  ±10–15% credible.
- **Proven inside the model:** strict conservation (0.000%), Carnot
  legality (0/36 violations), psat within 3% of reference.
- Carnot fractions 42–54% — defensible for oversized coils + ejector +
  EVI, and below what ideal-gas versions of this model printed.
- Still absent: CFD-calibrated coil maps, transient charge migration,
  in-tube refrigerant Δp (line drops are modeled; in-tube are lumped
  into UA), datasheet compressor maps, flip-actuator structural model,
  oil dynamics, two-phase entropy in quench mixing (clamped conservatively).
- Arctic ratings extrapolate property fits beyond the validated band;
  −50 °C feasibility is modeled, not measured.

## 25. Next Steps

1. REFPROP/CoolProp tables for the liquid side + two-phase mixtures.
2. Segment-wise ε-NTU coil discretization + in-tube refrigerant Δp.
3. Real compressor-map calibration (GMCC/Panasonic datasheet).
4. Room moisture balance (RH dynamics → latent feedback to controller).
5. Flip-actuator structural model: bearings, seals, rotary couplings.
6. R-290 variant + charge-limit safety model.
7. Blueprint → real CAD: tolerances, fasteners, materials, wiring
   schematic, pressure-vessel certification path.
