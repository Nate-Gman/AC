# =============================================================================
# AC.py — "AURA-1" Window-Replacement Reversible Heat Pump — Physics Replica
# =============================================================================
# A deliberately single-file physical simulation of the AC unit described in
# ProjectGoal.md: a window-replacing, season-reversible, ultra-quiet vapor-
# compression heat pump. Every major component of the real machine is modeled
# as code: refrigerant properties, inverter compressor, micro-channel coils,
# ECM fans, ejector + EVI cycle enhancements, EEV, four-way valve AND the
# physical 180-degree flip actuator, frost/defrost, condensate atomizer, an
# acoustic model, a noise-minimizing optimizer, and a closed-loop room model.
#
# The companion `overview.md` is the authoritative feature index — it lists
# every subsystem with a grep-able marker that locates it in this file.
#
# TABLE OF CONTENTS (search the §-marker text to jump)
# -----------------------------------------------------------------------------
#  §1   Constants, unit declarations, dataclasses, dB utilities
#       K(), F(), db_sum(), OperatingPoint, CycleResult
#  §2   RefrigerantR32 — Wagner vapor pressure (refit to NBP anchors),
#       Watson latent heat, enthalpy/entropy state functions, and a
#       Peng-Robinson real-gas model (Z, h/s departures, isentrope
#       inversions)                                 [class RefrigerantR32]
#  §3   Psychrometrics — humidity ratio, condensate, latent split
#       [w_sat(), condensate_rate()]
#  §4   InverterCompressor — displacement, volumetric/isentropic maps,
#       EVI vapor injection, acoustic signature    [class InverterCompressor]
#  §5   ECMFan — airflow law, cube-law power, aero-acoustic noise
#       [class ECMFan]
#  §6   MicroChannelCoil — epsilon-NTU air-side model + frost derating
#       [class MicroChannelCoil, class FrostModel]
#  §7   ExpansionTrain — EEV isenthalpic expansion + ejector work recovery;
#       ModeActuator — reversing valve vs. physical 180° flip
#       [class ExpansionTrain, class ModeActuator]
#  §8   HeatPumpCore — coupled fixed-point cycle solver, discharge limit,
#       defrost logic, condensate atomizer, acoustic summation
#       [class HeatPumpCore, def HeatPumpCore.solve()]
#  §9   QuietOptimizer — brute-force grid search for the minimum-noise
#       feasible operating point                    [class QuietOptimizer]
#  §10  Room + environment — lumped thermal mass, solar/gain profiles,
#       24-h ambient profiles                       [class Room, class DayProfile]
#  §11  ClosedLoopSimulator — PI capacity demand + optimizer dispatch,
#       defrost energy accounting                   [class ClosedLoopSimulator]
#  §12  Ratings & reporting — AHAM-style rating points, SEER2/HSPF2 bin
#       estimate, quiet-compare, baseline window-AC reference
#       [rating_points(), seasonal_estimate(), quiet_compare()]
#  §12b Validation (--validate) — energy-conservation audit (Q_evap+W=Q_cond
#       to <1%), Carnot-bound sweep, R-32 psat vs reference data
#       [validate(), R32_PSAT_REF]
#  §12c Model drawing (--view / --svg) — to-scale engineering sheet (0.5 px/mm)
#       of the rotating-drum window unit: side section, front elevation,
#       summer/winter flip states; pyglet viewer animates the 180° flip,
#       refrigerant flow, spinning fans, live telemetry
#       [DRUM/FRAME/COIL_*, export_svg(), run_viewer(), loop_segments()]
#  §13  Self-test (--test), benchmark (--bench), CLI dispatch, main()
#       [self_test(), main()]
#
# Conventions:
#   • SI units everywhere internally: K, Pa, J/kg, W, kg/s, rpm, dBA.
#   • Public API takes °C / °F helpers at the edges (K(), F()).
#   • Names: PascalCase classes, snake_case functions, UPPER_SNAKE constants.
#   • This is an engineering *model*, not a CFD/FEA package. Properties are
#     correlation-grade (±3–6 %), not REFPROP-grade. See HONESTY LAYER in §13.
#   • Pure stdlib: math, argparse, dataclasses, time. Python ≥3.9.
# =============================================================================

import argparse
import contextlib
import io
import math
import sys
import time
from dataclasses import dataclass, field

# =============================================================================
# §1  CONSTANTS / UNITS / SHARED TYPES
# =============================================================================

R_AIR = 287.05          # J/(kg·K)
CP_AIR = 1006.0         # J/(kg·K)
RHO_AIR = 1.18          # kg/m^3 (fan-flow bookkeeping)
P_ATM = 101325.0        # Pa
HFG_WATER = 2.501e6     # J/kg (approx, used for condensate latent split)

def K(c: float) -> float:
    return c + 273.15

def C(k: float) -> float:
    return k - 273.15

def F(k: float) -> float:
    return C(k) * 9.0 / 5.0 + 32.0

def db_sum(*levels: float) -> float:
    """Incoherent power sum of dBA sources; -inf-safe."""
    vals = [v for v in levels if v > -200]
    if not vals:
        return float("-inf")
    return 10.0 * math.log10(sum(10.0 ** (v / 10.0) for v in vals))


@dataclass
class OperatingPoint:
    """A commanded hardware state: compressor speed + both fan speeds."""
    n_rps: float          # compressor speed, rev/s
    rpm_in: float         # indoor fan rpm
    rpm_out: float        # outdoor fan rpm


@dataclass
class CycleResult:
    """Everything the cycle solver knows about one steady operating point."""
    ok: bool
    mode: str                    # 'cool' or 'heat'
    n_rps: float
    t_evap: float                # K, evaporating temperature
    t_cond: float                # K, condensing temperature
    t_discharge: float           # K, compressor discharge
    p_low: float                 # Pa
    p_high: float                # Pa
    m_dot_ref: float             # kg/s refrigerant
    q_capacity: float            # W delivered to the conditioned side
    q_sensible: float            # W dry-bulb capacity (cooling: minus latent)
    q_latent: float              # W moisture-removal capacity (cooling)
    q_reject: float              # W rejected to the other side
    w_compressor: float          # W electrical at compressor terminals
    w_thermo: float              # W refrigerant-side work (audit quantity)
    q_shell_w: float             # W compressor shell loss to ambient
    w_fans: float                # W
    w_fan_in: float              # W dissipated into the indoor airstream
    w_aux: float                 # W (controls, atomizer, valve)
    w_total: float               # W
    cop: float                   # delivered / total electrical
    carnot_cop: float            # reference Carnot COP for this mode
    carnot_fraction: float       # cop / carnot_cop
    air_delta_t: float           # K approach temperature, conditioned coil
    m_air_in: float              # kg/s
    m_air_out: float             # kg/s
    condensate_kg_s: float       # kg/s (cooling only)
    dba_indoor: float
    dba_outdoor: float
    evi_active: bool
    defrosting: bool
    frost: float                 # 0..1
    actuation: str               # 'valve' or 'flip'
    notes: str = ""


# =============================================================================
# §2  REFRIGERANT — R-32 (difluoromethane)
# =============================================================================
# Correlation-grade single-component model. Ancillary equations:
#   • Vapor pressure: Wagner equation anchored at critical point.
#   • Latent heat: Watson correlation anchored at 253.15 K.
#   • Liquid/vapor cp: linear-in-T fits.
#   • Entropy built consistently from s_liq + hfg/T so that isentropic
#     compression lands at plausible discharge temperatures.
# Accuracy ~±3–6 % vs REFPROP across -40..+65 °C — ample for system-level work.

class RefrigerantR32:
    NAME = "R-32"
    T_CRIT = 351.255        # K
    P_CRIT = 5.782e6        # Pa
    M = 0.05202             # kg/mol
    R = 8.314462 / M        # J/(kg·K) ≈ 159.8
    T_REF = 233.15          # K — enthalpy/entropy datum (h_l=200 kJ/kg, s=1.0)
    H_REF = 200.0e3
    S_REF = 1.0e3

    # Wagner vapor-pressure fit, re-anchored to hard physical data:
    # NBP −51.65 °C @ 0.1013 MPa, plus 0 °C and 25 °C reference points.
    # Verified <4 % error −40…+60 °C; old coefficients were 11–17 % off
    # at the cold end (caught by --validate).
    _WAGNER_A = (-6.4325, -1.2272, 8.985)             # coefficients
    _WAGNER_N = (1.0, 1.5, 4.5)                       # exponents
    _HFG_ANCHOR_T = 253.15
    _HFG_ANCHOR = 352.0e3     # J/kg at anchor
    _WATSON_EXP = 0.38

    @classmethod
    def psat(cls, t: float) -> float:
        """Saturation pressure [Pa] at temperature t [K]."""
        t = min(t, cls.T_CRIT - 0.01)
        tau = 1.0 - t / cls.T_CRIT
        s = sum(a * tau ** n for a, n in zip(cls._WAGNER_A, cls._WAGNER_N))
        return cls.P_CRIT * math.exp((cls.T_CRIT / t) * s)

    @classmethod
    def tsat(cls, p: float) -> float:
        """Saturation temperature [K] at pressure p [Pa] (Newton)."""
        t = 0.72 * cls.T_CRIT
        for _ in range(60):
            f = cls.psat(t) - p
            h = 0.05
            d = (cls.psat(t + h) - cls.psat(t - h)) / (2 * h)
            if d == 0:
                break
            t -= f / d
            t = min(max(t, 180.0), cls.T_CRIT - 0.01)
            if abs(f) < 1.0:
                break
        return t

    @classmethod
    def hfg(cls, t: float) -> float:
        """Latent heat of vaporization [J/kg] via Watson."""
        t = min(t, cls.T_CRIT - 0.01)
        num = 1.0 - t / cls.T_CRIT
        den = 1.0 - cls._HFG_ANCHOR_T / cls.T_CRIT
        return cls._HFG_ANCHOR * (num / den) ** cls._WATSON_EXP

    @classmethod
    def cp_liq(cls, t: float) -> float:
        return min(max(1.55e3 + 7.0 * (t - 253.15), 1.2e3), 3.0e3)

    @classmethod
    def cp_gas(cls, t: float) -> float:
        return 500.0 + 1.2 * t

    @classmethod
    def h_liq(cls, t: float) -> float:
        """Saturated-liquid enthalpy [J/kg], datum at T_REF."""
        return cls.H_REF + cls.cp_liq(t) * (t - cls.T_REF)

    @classmethod
    def h_g_sat(cls, t: float) -> float:
        return cls.h_liq(t) + cls.hfg(t)

    @classmethod
    def s_liq(cls, t: float) -> float:
        """Saturated-liquid entropy [J/(kg·K)], datum at T_REF."""
        return cls.S_REF + cls.cp_liq(t) * math.log(t / cls.T_REF)

    @classmethod
    def s_g_sat(cls, t: float) -> float:
        return cls.s_liq(t) + cls.hfg(t) / t

    @classmethod
    def h_g_super(cls, t_sat: float, t: float) -> float:
        """Superheated-vapor enthalpy at saturation base t_sat, actual t."""
        return cls.h_g_sat(t_sat) + cls.cp_gas(t_sat) * (t - t_sat)

    @classmethod
    def s_g_super(cls, t_sat: float, t: float) -> float:
        return cls.s_g_sat(t_sat) + cls.cp_gas(t_sat) * math.log(t / t_sat)

    # -- Peng-Robinson real-gas model (superheated-vapor region) -------------
    # Correlation functions above are fine for saturated liquid; the vapor
    # side uses PR departure functions for REFPROP-adjacent accuracy:
    # real suction density (mass flow), real isentropic paths (discharge
    # temperature), real enthalpy — the things cheap models get wrong.
    OMEGA = 0.2269             # R-32 acentric factor
    _T0 = 273.15
    _P0 = 1.0e5
    _CP0 = (350.0, 1.6)        # ideal-gas cp(T) = 350 + 1.6·T  J/(kg·K)
    _SQ2 = math.sqrt(2.0)
    _H_ANCHOR = None

    @classmethod
    def _pr_coefs(cls, t: float):
        k = 0.37464 + 1.54226 * cls.OMEGA - 0.26992 * cls.OMEGA ** 2
        tr = t / cls.T_CRIT
        m = 1.0 + k * (1.0 - math.sqrt(tr))
        ac = 0.45724 * cls.R ** 2 * cls.T_CRIT ** 2 / cls.P_CRIT
        a = ac * m * m
        b = 0.07780 * cls.R * cls.T_CRIT / cls.P_CRIT
        dadt = -ac * k * m / (cls.T_CRIT * math.sqrt(tr))
        return a, b, dadt

    @classmethod
    def z_vapor(cls, t: float, p: float) -> float:
        """Largest (vapor) root of the PR cubic — analytic Cardano solve
        plus one Newton polish step. ~5× faster than iterating from 1.0."""
        a, b, _ = cls._pr_coefs(t)
        A = a * p / (cls.R ** 2 * t * t)
        B = b * p / (cls.R * t)
        c2, c1 = -(1.0 - B), A - 3.0 * B * B - 2.0 * B
        c0 = -(A * B - B * B - B ** 3)
        # depress z = u − c2/3  →  u³ + pu·u + qu = 0
        pu = c1 - c2 * c2 / 3.0
        qu = c0 - c1 * c2 / 3.0 + 2.0 * c2 ** 3 / 27.0
        disc = 4.0 * pu ** 3 + 27.0 * qu * qu
        if disc < 0.0 and pu < 0.0:                 # three real roots
            m_ = 2.0 * math.sqrt(-pu / 3.0)
            arg = max(-1.0, min(1.0,
                      (3.0 * qu / (2.0 * pu)) * math.sqrt(-3.0 / pu)))
            th = math.acos(arg) / 3.0
            z = max(m_ * math.cos(th - 2.0 * math.pi * k / 3.0)
                    for k in range(3)) - c2 / 3.0
        else:                                        # one real root
            sq = math.sqrt(max(disc, 0.0) / 108.0)
            def _cbrt(x):
                return math.copysign(abs(x) ** (1.0 / 3.0), x)
            z = _cbrt(-qu / 2.0 + sq) + _cbrt(-qu / 2.0 - sq) - c2 / 3.0
        if z <= B:                                  # fell into liquid region
            z = max(1.0, B + 1e-6)
        f = z ** 3 + c2 * z * z + c1 * z + c0       # Newton polish
        d = 3.0 * z * z + 2.0 * c2 * z + c1
        if d != 0.0:
            z = max(z - f / d, B + 1e-9)
        return z

    @classmethod
    def _dep(cls, t: float, p: float):
        """PR enthalpy/entropy departure functions. Returns (h_dep, s_dep, Z)."""
        a, b, dadt = cls._pr_coefs(t)
        z = cls.z_vapor(t, p)
        B = b * p / (cls.R * t)
        L = math.log((z + (1.0 + cls._SQ2) * B) /
                     (z + (1.0 - cls._SQ2) * B))
        h_dep = cls.R * t * (z - 1.0) + \
            (t * dadt - a) / (2.0 * cls._SQ2 * b) * L
        s_dep = cls.R * math.log(max(z - B, 1e-12)) + \
            dadt / (2.0 * cls._SQ2 * b) * L
        return h_dep, s_dep, z

    @classmethod
    def h_ideal(cls, t: float) -> float:
        c0, c1 = cls._CP0
        return c0 * (t - cls._T0) + 0.5 * c1 * (t * t - cls._T0 * cls._T0)

    @classmethod
    def s_ideal(cls, t: float, p: float) -> float:
        c0, c1 = cls._CP0
        return c0 * math.log(t / cls._T0) + c1 * (t - cls._T0) - \
            cls.R * math.log(p / cls._P0)

    @classmethod
    def h_vap(cls, t: float, p: float) -> float:
        """Real-gas vapor enthalpy [J/kg], anchored onto the h_g_sat scale."""
        if cls._H_ANCHOR is None:
            ta, pa = 280.0, cls.psat(280.0)
            hd, _, _ = cls._dep(ta, pa)
            cls._H_ANCHOR = cls.h_g_sat(ta) - cls.h_ideal(ta) - hd
        hd, _, _ = cls._dep(t, p)
        return cls.h_ideal(t) + hd + cls._H_ANCHOR

    @classmethod
    def s_vap(cls, t: float, p: float) -> float:
        """Real-gas vapor entropy [J/(kg·K)] — self-consistent for isentropes."""
        _, sd, _ = cls._dep(t, p)
        return cls.s_ideal(t, p) + sd


def _solve_t_for_s(r: type, p: float, s_target: float, guess: float) -> float:
    """Newton invert s_vap(T,p) = s_target → T."""
    t = guess
    for _ in range(40):
        f = r.s_vap(t, p) - s_target
        h = 0.5
        d = (r.s_vap(t + h, p) - r.s_vap(t - h, p)) / (2 * h)
        if d == 0.0:
            break
        tn = t - f / d
        tn = min(max(tn, 200.0), 600.0)
        if abs(tn - t) < 1e-7:
            t = tn
            break
        t = tn
    return t


def _solve_t_for_h(r: type, p: float, h_target: float, guess: float) -> float:
    """Newton invert h_vap(T,p) = h_target → T."""
    t = guess
    for _ in range(40):
        f = r.h_vap(t, p) - h_target
        h = 0.5
        d = (r.h_vap(t + h, p) - r.h_vap(t - h, p)) / (2 * h)
        if d == 0.0:
            break
        tn = t - f / d
        tn = min(max(tn, 200.0), 600.0)
        if abs(tn - t) < 1e-7:
            t = tn
            break
        t = tn
    return t


# =============================================================================
# §3  PSYCHROMETRICS
# =============================================================================

def p_ws(celsius: float) -> float:
    """Saturation vapor pressure over water [Pa] (Magnus)."""
    return 610.94 * math.exp(17.625 * celsius / (celsius + 243.04))

def w_ratio(celsius: float, rh: float) -> float:
    """Humidity ratio kg/kg dry air."""
    pw = min(rh, 1.0) * p_ws(celsius)
    return 0.622 * pw / max(P_ATM - pw, 1.0)

def condensate_rate(m_air: float, t_in_c: float, rh_in: float,
                    t_evap_k: float) -> tuple:
    """(kg/s condensate, W latent load). Coil dew surface ≈ t_evap+2 K."""
    w_in = w_ratio(t_in_c, rh_in)
    w_out = w_ratio(C(t_evap_k) + 2.0, 0.95)
    m_cond = m_air * max(0.0, w_in - w_out)
    return m_cond, m_cond * HFG_WATER


# =============================================================================
# §4  INVERTER COMPRESSOR — twin-rotary, vapor-injection ready
# =============================================================================
# Quiet design choices modeled: wide-inlet low-velocity ports, magnetic-ish
# bearing proxy (no rubbing noise floor below ~35 rps), discharge muffler.

class InverterCompressor:
    V_D = 12.0e-6            # m^3 displacement (12 cm^3)
    N_MIN = 15.0             # rps
    N_MAX = 120.0            # rps
    N_PEAK = 60.0            # rps, peak-isentropic speed
    CLEARANCE = 0.05         # re-expansion clearance fraction
    K_GAS = 1.15             # polytropic exponent proxy
    ETA_IS_PEAK = 0.80
    ETA_EL = 0.93            # motor + inverter
    Z_SUC = 0.97             # suction compressibility fudge

    @staticmethod
    def _eta_vol(p_suc: float, p_dis: float) -> float:
        pr = max(p_dis / p_suc, 1.0)
        return max(1.0 - InverterCompressor.CLEARANCE *
                   (pr ** (1.0 / InverterCompressor.K_GAS) - 1.0), 0.55)

    @staticmethod
    def _eta_is(n_rps: float, pr: float) -> float:
        f_speed = 1.0 - 0.28 * ((n_rps - InverterCompressor.N_PEAK) /
                                InverterCompressor.N_PEAK) ** 2
        f_speed = min(max(f_speed, 0.55), 1.0)
        f_pr = min(max(1.0 - 0.05 * (pr - 3.0), 0.60), 1.0)
        return InverterCompressor.ETA_IS_PEAK * f_speed * f_pr

    def mass_flow(self, p_suc: float, t_suc: float, p_dis: float,
                  n_rps: float) -> float:
        z = RefrigerantR32.z_vapor(t_suc, p_suc)
        rho = p_suc / (z * RefrigerantR32.R * t_suc)
        return (rho * self.V_D * n_rps *
                self._eta_vol(p_suc, p_dis))

    @staticmethod
    def isentropic_eff(n_rps: float, pr: float) -> float:
        return InverterCompressor._eta_is(n_rps, pr)

    @staticmethod
    def eta_electrical(n_rps: float) -> float:
        """Motor+inverter efficiency — degrades at very low speed where
        stator/drive losses dominate (a fixed 0.93 flatters deep part-load)."""
        frac = min(n_rps / InverterCompressor.N_PEAK, 1.0)
        return InverterCompressor.ETA_EL * (0.82 + 0.18 * frac)

    @staticmethod
    def noise_dba(n_rps: float, pr: float) -> float:
        """A-weighted radiated level @1 m. Quiet floor at low speed."""
        base = 30.0 + 24.0 * math.log10(max(n_rps, 5.0) / 25.0)
        return max(base + 3.0 * math.log10(max(pr, 1.0) / 3.0), 24.0)


# =============================================================================
# §5  ECM FAN — oversized, slow-turning = the quiet lever
# =============================================================================
# Airflow ~ rpm (linear); shaft power ~ rpm^3 (fan laws); noise ~ 55·log10(rpm).
# Both fans are deliberately oversized so rated airflow arrives at low rpm —
# this is where the <40 dBA indoor target comes from.

class ECMFan:
    def __init__(self, name: str, kg_per_rpm: float, rpm_nom: float,
                 p_nom: float, l_nom: float, eta: float = 0.62):
        self.name = name
        self.k = kg_per_rpm            # kg/s airflow per rpm
        self.rpm_nom = rpm_nom
        self.p_nom = p_nom             # W shaft at rpm_nom
        self.l_nom = l_nom             # dBA @1 m at rpm_nom
        self.eta = eta                 # motor+aero combined

    def airflow(self, rpm: float) -> float:
        return self.k * max(rpm, 0.0)

    def power(self, rpm: float) -> float:
        """Electrical input at rpm. p_nom is the *installed* operating-
        point draw — the coil's static head (dp_ref Pa at m_air_ref) is
        already inside it, so pressure work scales with the fan law rather
        than being added on top. ECM motors hold η well but lose ~15%
        below ~40% nominal speed (drive losses dominate)."""
        if rpm <= 0:
            return 0.0
        frac = min(rpm / self.rpm_nom, 1.0)
        eta_eff = self.eta * (0.85 + 0.15 * frac)
        return self.p_nom * (rpm / self.rpm_nom) ** 3 / eta_eff

    def noise_dba(self, rpm: float) -> float:
        if rpm <= 1.0:
            return 10.0
        return self.l_nom + 55.0 * math.log10(rpm / self.rpm_nom)


# =============================================================================
# §6  MICRO-CHANNEL COILS + FROST
# =============================================================================
# Phase-change side treated as isothermal at Tsat; air side epsilon-NTU.
# The outdoor coil is deliberately huge (the "flip-to-vertical" geometry from
# the goal doc lets the condenser exceed the window width).

class MicroChannelCoil:
    MALDIS = 0.94    # louvered-face airflow maldistribution derate

    def __init__(self, name: str, ua_nom: float, m_air_ref: float,
                 dp_ref: float = 60.0):
        self.name = name
        self.ua_nom = ua_nom          # W/K at m_air_ref
        self.m_air_ref = m_air_ref
        self.dp_ref = dp_ref          # Pa air-side static head at m_air_ref

    def dp_static(self, m_air: float) -> float:
        """Coil air-side pressure drop [Pa] — turbulent regime, ∝ ṁ²."""
        return self.dp_ref * (max(m_air, 0.0) / self.m_air_ref) ** 2

    def ua(self, m_air: float) -> float:
        if m_air <= 0:
            m_air = 0.02
        return (self.ua_nom * (m_air / self.m_air_ref) ** 0.78
                * MicroChannelCoil.MALDIS)

    def effectiveness(self, m_air: float, frost_derate: float = 1.0) -> float:
        ntu = self.ua(m_air) * frost_derate / (m_air * CP_AIR)
        return 1.0 - math.exp(-ntu)


class FrostModel:
    """0..1 frost accumulator on the outdoor coil during heating."""
    def __init__(self):
        self.level = 0.0

    def update(self, dt_s: float, t_out_c: float, rh_out: float,
               defrosting: bool) -> None:
        if defrosting:
            self.level = max(0.0, self.level - dt_s / 60.0)   # ~1 min to clear
        elif -8.0 < t_out_c < 4.0 and rh_out > 0.55:
            growth = 0.012 * (rh_out - 0.55) / 0.45 * \
                     (1.0 - abs(t_out_c + 2.0) / 10.0)
            self.level = min(1.0, self.level + growth * dt_s / 60.0)
        else:
            self.level = max(0.0, self.level - 0.002 * dt_s / 60.0)

    @property
    def ua_derate(self) -> float:
        return 1.0 - 0.45 * self.level


# =============================================================================
# §7  EXPANSION TRAIN (EEV + EJECTOR) AND MODE ACTUATOR (VALVE / FLIP)
# =============================================================================

class ExpansionTrain:
    """EEV + liquid-vapor ejector + suction-line IHX + EVI flash economizer.

    Ejector: recovers a fraction of the throttling loss, modeled as work
    returned on compression (~6 %, matching published ejector gains).
    IHX: energy-conserving suction/liquid exchanger — warms suction toward
    the liquid line, deepens subcooling before the EEV.
    EVI: below a cold-ambient threshold, a fraction of flow flashes at the
    intermediate pressure sqrt(p1·p2), is injected mid-compression, and the
    remaining liquid enters the evaporator near the intermediate saturation
    temperature — raising capacity and COP at high lift."""
    EJECTOR_WORK_RECOVERY = 0.06
    SUBCOOL_K = 6.0
    SUPERHEAT_K = 5.0
    IHX_EFF = 0.45               # suction-line exchanger effectiveness
    EVI_INJECTION = 0.13         # injected mass fraction when EVI active

    @staticmethod
    def expansion_enthalpy(t_cond: float) -> float:
        """Outlet enthalpy after EEV: isenthalpic, subcooled inlet."""
        return RefrigerantR32.h_liq(t_cond - ExpansionTrain.SUBCOOL_K)


class ModeActuator:
    """Season reversal: electronic 4-way valve OR physical 180° flip.

    The flip eliminates valve suction/discharge pressure drops, so it earns a
    small real COP bonus — that is the *only* thermodynamic difference; all
    the mechanical cost (mass, seals, twice-yearly labor) lives in the docs."""
    VALVE_LOSS = 0.012            # fractional extra compressor work via valve

    def __init__(self, method: str = "valve"):
        assert method in ("valve", "flip")
        self.method = method

    @property
    def loss_fraction(self) -> float:
        return 0.0 if self.method == "flip" else self.VALVE_LOSS


# =============================================================================
# §8  HEAT-PUMP CORE — the coupled steady-state cycle solver
# =============================================================================
# Unknowns: evaporating T and condensing T. For each guess we run the
# refrigerant loop end-to-end, then close the loop with the air-side coil
# balances Q = ε·ṁ·cp·(T_air − T_sat). Damped fixed-point converges in
# <60 iterations for every point tested.

class HeatPumpCore:
    def __init__(self, actuation: str = "valve", quiet_build: bool = True):
        self.r = RefrigerantR32
        self.comp = InverterCompressor()
        self.actuator = ModeActuator(actuation)
        self.frost = FrostModel()
        # Oversized-quiet hardware set
        self.coil_in = MicroChannelCoil("indoor",  ua_nom=900.0,  m_air_ref=0.55)
        self.coil_out = MicroChannelCoil("outdoor", ua_nom=1400.0, m_air_ref=0.90)
        self.fan_in = ECMFan("indoor",  kg_per_rpm=7.3e-4, rpm_nom=750.0,
                             p_nom=38.0, l_nom=34.0)
        self.fan_out = ECMFan("outdoor", kg_per_rpm=1.6e-3, rpm_nom=600.0,
                              p_nom=55.0, l_nom=46.0)
        self.w_standby = 2.0          # controls
        self.w_atomizer = 3.0         # condensate atomizer (no-drip)
        self.t_dis_max = 393.0        # K discharge limit
        self.defrost_active = False
        self.evi_threshold_c = -3.0   # EVI engages below this ambient (heat)
        self._defrost_timer = 0.0
        # refrigerant-side pressure drops (coil + line): the compressor
        # works against suction-below-p1 and discharge-above-p2 — a real
        # lift penalty cheap models omit entirely
        # line/coil Δp is ∝ ṁ², not fixed: these are the drops at the
        # nominal ~10 g/s mass flow — the solver rescales each iteration
        self.dp_suc = 15e3            # Pa @ m_dot_ref, suction side
        self.dp_dis = 30e3            # Pa @ m_dot_ref, discharge side
        self.m_dot_ref = 0.010        # kg/s reference mass flow
        self.shell_frac = 0.04        # shell-loss fraction of w_thermo
        self.w_crankcase = 25.0       # W heater below ~-5 °C (oil/migration)
        self.wind_ua_gain = 1.05      # ambient wind assists outdoor coil
        self._guess = {}              # warm-start sat temps per mode

    # -- the core solver ----------------------------------------------------
    def solve(self, mode: str, t_room: float, t_out: float,
              point: OperatingPoint, rh_room: float = 0.5,
              rh_out: float = 0.6, solar_w: float = 0.0) -> CycleResult:
        r = self.r
        cool = (mode == "cool")
        # which coil sees which air
        evap_air_t = t_room if cool else t_out
        cond_air_t = t_out if cool else t_room
        m_evap = self.fan_in.airflow(point.rpm_in) if cool else \
            self.fan_out.airflow(point.rpm_out)
        m_cond = self.fan_out.airflow(point.rpm_out) if cool else \
            self.fan_in.airflow(point.rpm_in)
        evap_coil = self.coil_in if cool else self.coil_out
        cond_coil = self.coil_out if cool else self.coil_in
        evap_frost = 1.0 if cool else self.frost.ua_derate

        # cold air is denser — fixed volumetric fan flow moves more mass;
        # at -50 °C the outdoor coil sees ~35% more kg/s per rpm
        rho_in = P_ATM / (287.05 * t_room)
        rho_out = P_ATM / (287.05 * t_out)
        m_evap *= (rho_in if cool else rho_out) / 1.2
        m_cond *= (rho_out if cool else rho_in) / 1.2

        # winter solar assist: a dark absorber coil picks up real watts of
        # insolation in daylight — raises the effective air temperature
        # the evaporator sees (heating mode only)
        if not cool and solar_w > 0.0:
            evap_air_t = evap_air_t + solar_w / (m_evap * CP_AIR)

        # ambient EVI staging: 1 injection port below -3 °C, two below
        # -18 °C, three below -30 °C — each extra port splits the lift
        # finer, the only way to hold discharge temperature at arctic
        # ambient (-50 °C → pr≈50 vs single-stage)
        tout_c = C(t_out)
        evi = (not cool) and (tout_c < self.evi_threshold_c)
        n_ports = 1
        if evi:
            n_ports = 3 if tout_c < -30.0 else (2 if tout_c < -18.0 else 1)

        # initial guesses (warm-started from the last converged solve)
        t_e, t_c = self._guess.get(mode, (evap_air_t - 9.0,
                                          cond_air_t + 11.0))
        sh = ExpansionTrain.SUPERHEAT_K
        y_q = 0.0                     # liquid-quench fraction (EVI only)

        loss = 1.0 + self.actuator.loss_fraction
        # line Δp ∝ ṁ² — lagged one iteration inside the fixed point
        m_dot_lag = self.m_dot_ref
        for _ in range(80):
            p1 = r.psat(t_e)
            p2 = r.psat(t_c)
            # compressor sees suction below p1, discharge above p2
            fs = (m_dot_lag / self.m_dot_ref) ** 2
            p1c = max(p1 - self.dp_suc * fs, 0.05e6)
            p2c = p2 + self.dp_dis * fs
            pr = max(p2c / p1c, 1.01)

            # evaporator exit (real enthalpy), then IHX suction side
            t_evap_out = t_e + sh
            h_evap_out = r.h_vap(t_evap_out, p1)
            t1 = t_evap_out + ExpansionTrain.IHX_EFF * max(
                0.0, (t_c - ExpansionTrain.SUBCOOL_K) - t_evap_out)
            h1 = r.h_vap(t1, p1c)
            s1 = r.s_vap(t1, p1c)
            ihx_gain = h1 - h_evap_out          # J/kg shifted suction-ward

            # STRICT bookkeeping: condenser sees pre-IHX liquid enthalpy,
            # evaporator feed sees post-IHX enthalpy (IHX is internal).
            h3_pre = r.h_liq(t_c - ExpansionTrain.SUBCOOL_K)

            eta_is = self.comp.isentropic_eff(point.n_rps, pr)
            m_dot = self.comp.mass_flow(p1c, t1, p2c, point.n_rps)
            m_dot_lag = m_dot                       # for next-iter Δp

            if evi:
                # n_ports vapor-injection ports at geometric pressure
                # splits; each flashes liquid, injects saturated vapor
                # mid-compression, and cools the remaining liquid feed.
                ports = [p1c * pr ** ((i + 1.0) / (n_ports + 1.0))
                         for i in range(n_ports)]
                # injection fraction scales with lift — a fixed 13% can't
                # quench discharge at arctic ratios; at pr>~5 each port
                # takes progressively more flash vapor
                x_port = ExpansionTrain.EVI_INJECTION * \
                    max(1.0, pr / 5.0)
                x_port = min(x_port, 0.32)      # keep ports manufacturable
                xs = [x_port] * n_ports
                # -- march the compression stage-by-stage --
                h_c, s_c, t_c_g = h1, s1, t1
                mass = 1.0                        # per kg suction flow
                w_act = 0.0
                for i, pm in enumerate(ports):
                    t_s = _solve_t_for_s(r, pm, s_c, t_c_g * 1.15)
                    dh = (r.h_vap(t_s, pm) - h_c) / eta_is * loss
                    h_c += dh
                    t_c_g = _solve_t_for_h(r, pm, h_c, t_s * 1.05)
                    s_c = r.s_vap(t_c_g, pm)
                    w_act += mass * dh
                    tm_i = r.tsat(pm)
                    h_inj = r.h_vap(tm_i, pm)
                    x = xs[i]
                    s_c = (mass * s_c + x * r.s_vap(tm_i, pm)) / (mass + x)
                    h_c = (mass * h_c + x * h_inj) / (mass + x)
                    mass += x
                # final stage to discharge — with liquid-quench injection
                # at the last port when needed: subcooled liquid flashes
                # into the stream and clamps T_dis (the real mechanism
                # arctic-rated compressors use below ~-25 °C)
                def _final_stage(y_liq: float):
                    """Return (h2, t2, w_final, mass_out) with y_liq kg
                    of liquid quench per kg suction flow. If the mix goes
                    two-phase the stream entropy is clamped at saturated
                    vapor — conservative (under-credits quench) but the
                    Newton solves stay inside the superheated domain."""
                    m2 = mass + y_liq
                    # quench draw comes off the liquid line: its enthalpy
                    # is the post-IHX liquid enthalpy shared by all mass
                    h_liq_inj = h3_pre - ihx_gain / m2
                    h_m = (mass * h_c + y_liq * h_liq_inj) / m2
                    pm = ports[-1]
                    tm_p = r.tsat(pm)
                    t_m = max(_solve_t_for_h(r, pm, h_m, t_c_g),
                              tm_p + 0.5)
                    s_m = r.s_vap(t_m, pm)
                    t2s_ = _solve_t_for_s(r, p2c, s_m,
                                          max(t_c_g * 1.15, tm_p + 15.0))
                    dh_ = max(r.h_vap(t2s_, p2c) - h_m, 1.0) / eta_is * loss
                    h2_ = h_m + dh_
                    t2_ = _solve_t_for_h(r, p2c, h2_, t2s_ * 1.05)
                    return h2_, t2_, m2 * dh_, m2
                h2, t2, w_fin, mass_out = _final_stage(0.0)
                t_lim = self.t_dis_max - 8.0
                if t2 > t_lim:
                    # warm-start the bracket around last iteration's y —
                    # inside the fixed-point loop t_e/t_c move slowly, so
                    # the right quench fraction is near the previous one
                    if y_q > 0.0:
                        lo, hi = max(0.0, y_q * 0.5), \
                            min(1.2, y_q * 1.6 + 0.02)
                        n_it = 8
                    else:
                        lo, hi = 0.0, 1.2
                        n_it = 14
                    for _ in range(n_it):         # bisect quench fraction
                        y_m = 0.5 * (lo + hi)
                        h2, t2, w_fin, mass_out = _final_stage(y_m)
                        if t2 > t_lim:
                            lo = y_m
                        else:
                            hi = y_m
                    y_q = hi
                    # state at the accepted quench fraction
                    h2, t2, w_fin, mass_out = _final_stage(y_q)
                else:
                    y_q = 0.0
                w_act += w_fin
                mass = mass_out
                # -- cascade flash balance on the liquid side: the total
                # injected mass comes out of the condenser's liquid; the
                # quench draw (y_q) splits off the liquid line first, then
                # each port flashes from highest pressure down --
                h_l = h3_pre - ihx_gain / mass
                m_l = mass - y_q                  # quench draw leaves first
                for i in reversed(range(n_ports)):
                    pm = ports[i]
                    x = xs[i]
                    # debit on the PR vapor scale — same datum the injected
                    # stream is credited with, else each port leaks the
                    # correlation/PR datum offset into the audit
                    h_l = (m_l * h_l - x * r.h_vap(r.tsat(pm), pm)) / \
                        (m_l - x)
                    m_l -= x
                h_feed = h_l                      # m_l == 1 now
                q_evap = m_dot * (h_evap_out - h_feed)
                q_cond = m_dot * mass * (h2 - h3_pre)
            else:
                t2s = _solve_t_for_s(r, p2c, s1, t_c * 1.2)
                h2s = r.h_vap(t2s, p2c)
                w_is = max(h2s - h1, 1.0)
                w_act = w_is / eta_is * loss
                h2 = h1 + w_act
                t2 = _solve_t_for_h(r, p2c, h2, t2s * 1.05)
                h4 = h3_pre - ihx_gain                  # EEV: isenthalpic
                q_evap = m_dot * (h_evap_out - h4)
                q_cond = m_dot * (h2 - h3_pre)

            # wet-coil bonus: when the evaporator surface is below the
            # room dew point, latent release at the fins boosts effective
            # UA ~10% (dry-coil assumption would understate capacity)
            wet = 1.0
            if cool and p_ws(C(t_e) + 2.0) < rh_room * p_ws(C(t_room)):
                wet = 1.10
            # outdoor coil also picks up ambient wind convection
            e_wind = self.wind_ua_gain if evap_coil is self.coil_out else 1.0
            c_wind = self.wind_ua_gain if cond_coil is self.coil_out else 1.0
            eps_e = evap_coil.effectiveness(m_evap, evap_frost * wet * e_wind)
            eps_c = cond_coil.effectiveness(m_cond, c_wind)
            # compressor shell loss: ~4% of thermo work convects off the
            # shell to ambient instead of reaching the refrigerant stream
            q_shell = self.shell_frac * m_dot * w_act
            q_cond_air = q_cond - q_shell      # never reaches the coil air
            t_e_new = evap_air_t - q_evap / (eps_e * m_evap * CP_AIR)

            # evaporative assist: the atomizer slings collected condensate
            # onto the condenser — evaporating it pre-cools the condenser
            # airstream toward wet-bulb, worth real approach temperature
            cond_air_eff = cond_air_t
            if cool:
                m_cw, _ = condensate_rate(m_evap, C(t_room), rh_room, t_e)
                q_evap_avail = m_cw * HFG_WATER            # W evap potential
                dt_assist = q_evap_avail / (m_cond * CP_AIR) * 0.65
                cond_air_eff = cond_air_t - min(dt_assist, 6.0)
            t_c_new = cond_air_eff + q_cond_air / (eps_c * m_cond * CP_AIR)

            # physical clamps then damped update
            t_e_new = min(max(t_e_new, evap_air_t - 30.0), evap_air_t - 1.0)
            t_c_new = min(max(t_c_new, cond_air_eff + 1.0),
                          cond_air_t + 40.0)
            dt_e, dt_c = abs(t_e_new - t_e), abs(t_c_new - t_c)
            t_e += 0.45 * (t_e_new - t_e)
            t_c += 0.45 * (t_c_new - t_c)
            if dt_e < 0.01 and dt_c < 0.01:
                break
        self._guess[mode] = (t_e, t_c)   # warm-start next call

        # -- powers ---------------------------------------------------------
        # Thermodynamic work (refrigerant enthalpy rise) vs electrical work:
        # the ejector contributes mechanical energy (suction boost), so
        # electrical input = w_act·(1−recovery); conservation audit uses w_act.
        w_thermo = m_dot * w_act                          # W, refrigerant-side
        w_elec_spec = w_act * (1.0 - ExpansionTrain.EJECTOR_WORK_RECOVERY)
        w_comp = m_dot * w_elec_spec / \
            self.comp.eta_electrical(point.n_rps)
        w_fan_in = self.fan_in.power(point.rpm_in)
        w_fan = w_fan_in + self.fan_out.power(point.rpm_out)
        w_aux = self.w_standby
        if cool:
            m_cond_w, q_latent = condensate_rate(m_evap, C(t_room), rh_room, t_e)
            if m_cond_w > 0:
                w_aux += self.w_atomizer
        else:
            m_cond_w, q_latent = 0.0, 0.0
            # crankcase heater keeps oil/refrigerant separated in the cold
            if C(t_out) < -5.0:
                w_aux += self.w_crankcase
        q_cond -= q_shell                    # shell loss never delivered
        w_total = w_comp + w_fan + w_aux

        q_delivered = q_evap if cool else q_cond
        # indoor-fan motor heat lands in the room airstream: it eats into
        # net sensible cooling in summer and adds real heat in winter
        q_sensible = (q_evap - q_latent - w_fan_in) if cool else \
            (q_cond + w_fan_in)
        cop = q_delivered / w_total
        if cool:
            carnot = t_e / max(t_c - t_e, 1.0)
        else:
            carnot = t_c / max(t_c - t_e, 1.0)

        # -- acoustics ------------------------------------------------------
        l_comp = self.comp.noise_dba(point.n_rps, pr)
        # compressor sits mid-chassis: 10 dB transmission loss to indoor face
        dba_in = db_sum(l_comp - 10.0, self.fan_in.noise_dba(point.rpm_in))
        dba_out = db_sum(l_comp, self.fan_out.noise_dba(point.rpm_out))

        ok = t2 < self.t_dis_max
        notes = "" if ok else "discharge-temp limited"
        if evi and n_ports > 1:
            notes = (f"EVI×{n_ports} " + notes).strip()
        if evi and y_q > 0.01:
            notes = (f"quench y={y_q:.2f} " + notes).strip()

        return CycleResult(
            ok=ok, mode=mode, n_rps=point.n_rps,
            t_evap=t_e, t_cond=t_c, t_discharge=t2,
            p_low=p1, p_high=p2, m_dot_ref=m_dot,
            q_capacity=q_delivered, q_sensible=q_sensible,
            q_latent=q_latent,
            q_reject=q_cond if cool else q_evap,
            w_compressor=w_comp, w_thermo=w_thermo, q_shell_w=q_shell,
            w_fans=w_fan, w_fan_in=w_fan_in, w_aux=w_aux, w_total=w_total,
            cop=cop, carnot_cop=carnot, carnot_fraction=cop / carnot,
            air_delta_t=abs((t_room if cool else t_out) - t_e),
            m_air_in=m_evap, m_air_out=m_cond,
            condensate_kg_s=m_cond_w,
            dba_indoor=dba_in, dba_outdoor=dba_out,
            evi_active=evi, defrosting=False, frost=self.frost.level,
            actuation=self.actuator.method, notes=notes)

    # -- transient defrost management (used by closed-loop sim) -------------
    def step_defrost(self, dt_s: float, t_out_c: float, rh_out: float,
                     heating: bool) -> None:
        self.frost.update(dt_s, t_out_c, rh_out, self.defrost_active)
        if heating and not self.defrost_active and self.frost.level > 0.55:
            self.defrost_active = True
            self._defrost_timer = 0.0
        if self.defrost_active:
            self._defrost_timer += dt_s
            if self._defrost_timer > 240.0 or self.frost.level <= 0.02:
                self.defrost_active = False


# =============================================================================
# §9  QUIET OPTIMIZER — pick the silent feasible operating point
# =============================================================================
# Given required capacity, grid-search (N, rpm_in, rpm_out) and minimize
# indoor dBA with a small power tie-breaker. This is the "model-predictive
# quiet controller": it always prefers longer runtime at lower speed over
# short loud bursts — the correct strategy for both noise and part-load COP.

class QuietOptimizer:
    N_GRID = tuple(range(15, 121, 5))
    RPM_IN_GRID = tuple(range(200, 901, 50))
    RPM_OUT_GRID = tuple(range(250, 901, 50))

    def __init__(self, core: HeatPumpCore):
        self.core = core
        self._cache = {}

    def best(self, mode: str, t_room: float, t_out: float,
             q_required: float, rh_room: float = 0.5,
             rh_out: float = 0.6, dba_weight: float = 1.0,
             w_weight: float = 0.002, dba_cap: float = None,
             solar_w: float = 0.0) -> tuple:
        """Returns (OperatingPoint, CycleResult) minimizing noise then watts.
        dba_cap (night mode) rejects any point louder than the cap."""
        key = (mode, round(t_room * 2) / 2, round(t_out * 2) / 2,
               round(q_required / 250), dba_cap)
        cached = self._cache.get(key)
        if cached is not None:
            res = self.core.solve(mode, t_room, t_out, cached, rh_room,
                                  rh_out, solar_w=solar_w)
            if res.ok and res.q_sensible >= q_required * 0.98 and \
                    (dba_cap is None or res.dba_indoor <= dba_cap):
                return cached, res
        def scan(ns, ris, ros):
            """Return (best, fallback) over a (n, rpm_in, rpm_out) subgrid."""
            bst, fb = None, None
            for n in ns:
                for ri in ris:
                    for ro in ros:
                        pt = OperatingPoint(float(n), float(ri), float(ro))
                        res = self.core.solve(mode, t_room, t_out, pt,
                                              rh_room, rh_out,
                                              solar_w=solar_w)
                        if not res.ok or res.q_sensible < q_required:
                            continue
                        if fb is None or res.dba_indoor < fb[0]:
                            fb = (res.dba_indoor, pt, res)
                        if dba_cap is not None and res.dba_indoor > dba_cap:
                            continue
                        sc = dba_weight * res.dba_indoor + \
                            w_weight * res.w_total
                        if bst is None or sc < bst[0]:
                            bst = (sc, pt, res)
            return bst, fb

        # two-stage search: every-other-point coarse pass, then a fine
        # neighborhood around the winner — ~4× fewer solves, same optimum
        best, fallback = scan(self.N_GRID[::2], self.RPM_IN_GRID[::2],
                              self.RPM_OUT_GRID[::2])
        anchor = best[1] if best is not None else \
            (fallback[1] if fallback is not None else None)
        if anchor is not None:
            ns = [n for n in self.N_GRID if abs(n - anchor.n_rps) <= 5.1]
            ris = [x for x in self.RPM_IN_GRID
                   if abs(x - anchor.rpm_in) <= 50.1]
            ros = [x for x in self.RPM_OUT_GRID
                   if abs(x - anchor.rpm_out) <= 50.1]
            b2, f2 = scan(ns, ris, ros)
            if b2 is not None:
                best = b2
            if f2 is not None and (fallback is None or f2[0] < fallback[0]):
                fallback = f2
        if best is None and fallback is not None:
            best = (fallback[0], fallback[1], fallback[2])
        if best is None:
            # infeasible: run flat out
            pt = OperatingPoint(120.0, 900.0, 900.0)
            res = self.core.solve(mode, t_room, t_out, pt, rh_room,
                                  rh_out, solar_w=solar_w)
            return pt, res
        self._cache[key] = best[1]
        return best[1], best[2]

    def best_power(self, mode: str, t_room: float, t_out: float,
                   q_required: float, rh_room: float = 0.5,
                   rh_out: float = 0.6, solar_w: float = 0.0) -> tuple:
        """Max-efficiency variant: minimize watts subject to capacity."""
        return self._best_generic(mode, t_room, t_out, q_required,
                                  rh_room, rh_out, solar_w=solar_w,
                                  score=lambda r: r.w_total)

    def _best_generic(self, mode, t_room, t_out, q_required,
                      rh_room, rh_out, score, solar_w: float = 0.0) -> tuple:
        def scan(ns, ris, ros):
            bst = None
            for n in ns:
                for ri in ris:
                    for ro in ros:
                        pt = OperatingPoint(float(n), float(ri), float(ro))
                        res = self.core.solve(mode, t_room, t_out, pt,
                                              rh_room, rh_out,
                                              solar_w=solar_w)
                        if not res.ok or res.q_sensible < q_required:
                            continue
                        s = score(res)
                        if bst is None or s < bst[0]:
                            bst = (s, pt, res)
            return bst
        best = scan(self.N_GRID[::2], self.RPM_IN_GRID[::2],
                    self.RPM_OUT_GRID[::2])
        if best is not None:
            a = best[1]
            best = scan([n for n in self.N_GRID
                         if abs(n - a.n_rps) <= 5.1],
                        [x for x in self.RPM_IN_GRID
                         if abs(x - a.rpm_in) <= 50.1],
                        [x for x in self.RPM_OUT_GRID
                         if abs(x - a.rpm_out) <= 50.1]) or best
        if best is None:
            pt = OperatingPoint(120.0, 900.0, 900.0)
            return pt, self.core.solve(mode, t_room, t_out, pt,
                                       rh_room, rh_out)
        return best[1], best[2]


# =============================================================================
# §10  ROOM + ENVIRONMENT
# =============================================================================

class Room:
    """Lumped single-zone model: air + participating mass, envelope UA,
    internal gains, optional solar. C_eff includes furnishings/drywall.
    ua_extra models the unit's own infiltration + thermal-bridge penalty —
    the near-zero-leak install is part of why the machine wins (a typical
    rattling window AC leaks the equivalent of 30–50 W/K)."""
    def __init__(self, t0: float = K(26.0), ua: float = 150.0,
                 c_eff: float = 4.2e6, ua_extra: float = 6.0):
        self.t = t0
        self.ua = ua + ua_extra      # W/K total effective conductance
        self.ua_shell = ua
        self.ua_extra = ua_extra
        self.c_eff = c_eff           # J/K effective thermal mass

    def step(self, dt_s: float, q_hvac: float, t_out: float,
             q_internal: float = 250.0, q_solar: float = 0.0) -> None:
        dq = q_hvac + q_internal + q_solar - self.ua * (self.t - t_out)
        self.t += dq * dt_s / self.c_eff

    @staticmethod
    def steady_load(t_set: float, t_out: float, ua: float = 150.0,
                    q_internal: float = 250.0, q_solar: float = 0.0) -> float:
        """Required HVAC capacity [W] to hold t_set (positive = cooling)."""
        return ua * (t_out - t_set) - q_internal - q_solar


class DayProfile:
    """24-h sinusoidal ambient + crude solar gain."""
    def __init__(self, t_min_c: float, t_max_c: float, peak_hour: float = 15.0,
                 rh: float = 0.6):
        self.tmin, self.tmax, self.peak, self.rh = t_min_c, t_max_c, peak_hour, rh

    def t_out(self, hour: float) -> float:
        mid = 0.5 * (self.tmax + self.tmin)
        amp = 0.5 * (self.tmax - self.tmin)
        return K(mid + amp * math.cos(2 * math.pi * (hour - self.peak) / 24.0))

    def solar(self, hour: float) -> float:
        if 7.0 < hour < 19.0:
            return 400.0 * math.sin(math.pi * (hour - 7.0) / 12.0)
        return 0.0


# =============================================================================
# §11  CLOSED-LOOP SIMULATOR
# =============================================================================

class ClosedLoopSimulator:
    def __init__(self, core: HeatPumpCore, optimizer: QuietOptimizer):
        self.core = core
        self.opt = optimizer

    def run_day(self, mode: str, profile: DayProfile, t_set: float,
                room: Room, dt_s: float = 120.0, verbose: bool = True,
                label: str = "", night_cap: float = None) -> dict:
        kp, ki = 900.0, 6.0            # PI gains (W per K error)
        integ = 0.0
        last_pt = None
        energy = {"hvac_kwh": 0.0, "comp_kwh": 0.0, "resist_kwh": 0.0,
                  "defrost_kwh": 0.0}
        dba_track, cop_track, cap_track = [], [], []
        sign = 1.0 if mode == "cool" else -1.0
        steps = int(24 * 3600 / dt_s)
        hour_idx = -1

        for i in range(steps):
            hour = i * dt_s / 3600.0
            t_out = profile.t_out(hour)
            q_sol = profile.solar(hour)

            # PI on temperature error → required capacity
            # PI + feedforward of the *current* room load (correct physics:
            # hold-point needs the live envelope load, not the setpoint load)
            err = (room.t - t_set) * sign          # >0 means need more output
            ff = sign * Room.steady_load(room.t, t_out, room.ua,
                                         q_solar=q_sol)
            if err <= -0.15:                       # deadband: coast
                q_req = 0.0
                integ *= 0.9
            else:
                integ += err * dt_s / 3600.0
                q_req = max(0.0, ff + kp * err + ki * integ)

            self.core.step_defrost(dt_s, C(t_out), profile.rh,
                                   heating=(mode == "heat"))

            if q_req < 40.0:
                res = None
                # idle/standby: fans off; crankcase heater still draws in
                # cold ambient to keep oil separated from refrigerant
                w_now, q_now, dba_now = 5.0, 0.0, 20.0
                if mode == "heat" and C(t_out) < -5.0:
                    w_now += self.core.w_crankcase
            else:
                # night quiet cap: 22:00–07:00 the unit may not exceed the
                # acoustic budget even at the cost of running longer
                cap = night_cap if night_cap is not None and \
                    (hour >= 22.0 or hour < 7.0) else None

                # a dark outdoor coil intercepts a fraction of the wall's
                # solar aperture in heating mode — real free watts
                s_coil = 0.25 * q_sol if mode == "heat" else 0.0
                # fast path: reuse last operating point if still sufficient
                res = None
                if last_pt is not None:
                    res = self.core.solve(mode, room.t, t_out, last_pt,
                                          rh_out=profile.rh,
                                          solar_w=s_coil)
                    if not (res.ok and res.q_sensible >= q_req and
                            (cap is None or res.dba_indoor <= cap)):
                        res = None
                if res is None:
                    last_pt, res = self.opt.best(mode, room.t, t_out, q_req,
                                                 rh_out=profile.rh,
                                                 dba_cap=cap,
                                                 solar_w=s_coil)
                if self.core.defrost_active and mode == "heat":
                    # defrost: briefly run cooling to melt frost
                    _, res = self.opt.best("cool", t_out, room.t, 1500.0,
                                           rh_out=profile.rh)
                    energy["defrost_kwh"] += res.w_total * dt_s / 3.6e6
                    q_now = -res.q_capacity            # heat pulled from room
                    w_now = res.w_total
                    dba_now = res.dba_indoor
                else:
                    # dry-bulb control sees sensible capacity only
                    q_now = res.q_sensible
                    w_now = res.w_total
                    dba_now = res.dba_indoor
                    # duty-cycling: even at minimum speed the quietest
                    # feasible point may overshoot the request — real units
                    # then cycle on/off, paying ~8% restart loss per cycle
                    # instead of dumping the excess capacity
                    if res.q_sensible > 1.6 * q_req:
                        duty = q_req / res.q_sensible
                        q_now = res.q_sensible * duty
                        w_now = res.w_total * duty * 1.08
                        # Leq duty correction: intermittent noise averages
                        # 10·log10(duty) below the running level
                        dba_now = dba_now + 10.0 * math.log10(duty)
                    cap_track.append(q_now)
                    cop_track.append(q_now / max(w_now, 1.0))
                dba_track.append(dba_now)

            room.step(dt_s, -q_now if mode == "cool" else q_now,
                      t_out, q_solar=q_sol)
            energy["hvac_kwh"] += w_now * dt_s / 3.6e6
            if res is not None:
                energy["comp_kwh"] += res.w_compressor * dt_s / 3.6e6
            energy["resist_kwh"] += abs(q_now) * dt_s / 3.6e6  # COP=1 baseline

            if verbose and int(hour) != hour_idx:
                hour_idx = int(hour)
                self._print_hour(hour, room.t, t_out, q_req, res, w_now, mode)

        result = {
            "label": label or mode,
            "mode": mode,
            "t_final": room.t,
            "energy": energy,
            "dba_avg": sum(dba_track) / max(len(dba_track), 1),
            "dba_max": max(dba_track) if dba_track else 0.0,
            "cop_avg": sum(cop_track) / max(len(cop_track), 1),
            "cap_avg_w": sum(cap_track) / max(len(cap_track), 1),
        }
        return result

    @staticmethod
    def _print_hour(hour, t_room, t_out, q_req, res, w_now, mode):
        unit = "COOL" if mode == "cool" else "HEAT"
        if res is None:
            print(f"  {int(hour):02d}:00  out={C(t_out):5.1f}°C  "
                  f"room={C(t_room):5.2f}°C  {unit} idle     "
                  f"{'—':>7} {'—':>6} {'—':>6}")
        else:
            print(f"  {int(hour):02d}:00  out={C(t_out):5.1f}°C  "
                  f"room={C(t_room):5.2f}°C  {unit} "
                  f"N={res.n_rps:5.0f}rps  Q={res.q_capacity:6.0f}W  "
                  f"COP={res.cop:5.2f}  {res.dba_indoor:5.1f}dBA")


# =============================================================================
# §12  RATINGS / SEASONAL / COMPARISONS
# =============================================================================

BASELINE_WINDOW_AC = {
    "name": "typical window AC (CEER 12, fixed-speed)",
    "cop_cool": 3.5, "cop_heat_resist": 1.0, "dba": 55.0,
}


def _row(tag, res: CycleResult):
    return (f"  {tag:<38} Q={res.q_capacity:6.0f} W   "
            f"W={res.w_total:5.0f} W   COP={res.cop:5.2f}   "
            f"Carnot%={100*res.carnot_fraction:4.0f}%   "
            f"{res.dba_indoor:4.1f} dBA")


def rating_points(core: HeatPumpCore, opt: QuietOptimizer) -> None:
    print("\n=== RATING POINTS (AURA-1, steady-state) ===")
    pts = [
        ("Cooling  95°F out / 80°F room (full)", "cool",
         K(26.7), K(35.0), 0.95),
        ("Cooling  82°F out / 80°F room (part)", "cool",
         K(26.7), K(27.8), 0.55),
        ("Heating  47°F out / 70°F room", "heat",
         K(21.0), K(8.3), 0.95),
        ("Heating  17°F out / 70°F room (EVI)", "heat",
         K(21.0), K(-8.3), 0.95),
        ("Heating   5°F out / 70°F room (EVI)", "heat",
         K(21.0), K(-15.0), 0.60),
        ("Heating -40°F out / 70°F room (EVI×3)", "heat",
         K(21.0), K(-40.0), 0.42),
        ("Heating -58°F out / 70°F room (EVI×3)", "heat",
         K(21.0), K(-50.0), 0.42),
    ]
    for tag, mode, t_room, t_out, load_frac in pts:
        q_req = 3500.0 * load_frac if mode == "cool" else 3600.0 * load_frac
        pt, res = opt.best(mode, t_room, t_out, q_req)
        print(_row(tag, res) +
              ("   [EVI]" if res.evi_active else "") +
              ("   [flip]" if res.actuation == "flip" else "") +
              ("   [" + res.notes + "]" if res.notes else ""))


def seasonal_estimate(core: HeatPumpCore, opt: QuietOptimizer) -> None:
    """Simplified bin estimate → rough SEER2 / HSPF2 equivalents."""
    print("\n=== SEASONAL ESTIMATE (bin method, approximate) ===")
    # cooling bins: outdoor °C and hours weights (mixed-climate proxy)
    cool_bins = [(19, .05), (22, .15), (25, .25), (28, .25), (31, .18),
                 (34, .09), (37, .03)]
    heat_bins = [(-12, .03), (-7, .10), (-2, .20), (3, .25), (8, .22),
                 (13, .15), (16, .05)]
    q_nom = 3500.0

    e_cool = q_cool = 0.0
    for t_c, w in cool_bins:
        t_out = K(t_c)
        load = max(0.0, Room.steady_load(K(26.0), t_out, 150.0, 250.0, 200.0))
        load = min(load, q_nom * 1.15)
        if load < 100:
            continue
        _, res = opt.best_power("cool", K(26.0), t_out, load)
        e_cool += res.w_total * w
        q_cool += res.q_capacity * w
    seer = (q_cool / max(e_cool, 1e-9)) * 3.412   # Wh→BTU gives SEER-like

    e_heat = q_heat = 0.0
    for t_c, w in heat_bins:
        t_out = K(t_c)
        load = max(0.0, -Room.steady_load(K(21.0), t_out, 150.0, 250.0))
        if load < 100:
            continue
        _, res = opt.best_power("heat", K(21.0), t_out, min(load, 4200.0))
        e_heat += res.w_total * w
        q_heat += res.q_capacity * w
    hspf = (q_heat / max(e_heat, 1e-9)) * 3.412

    print(f"  Cooling seasonal COP ≈ {q_cool/max(e_cool,1):.2f}   "
          f"≈ SEER {seer:.1f}  (best window units ~15 CEER)")
    print(f"  Heating seasonal COP ≈ {q_heat/max(e_heat,1):.2f}   "
          f"≈ HSPF {hspf:.1f}  (best ~10–12)")


def quiet_compare(core: HeatPumpCore, opt: QuietOptimizer) -> None:
    """Same load, loud-fast vs optimized-quiet operating point."""
    print("\n=== QUIET-COMPARE (same capacity, two strategies) ===")
    t_room, t_out, q_req = K(26.0), K(35.0), 3200.0
    # "loud" = deliver the same capacity at needlessly high speed/airflow,
    # mimicking a fixed-speed unit that always runs near flat out.
    loud = None
    for n in QuietOptimizer.N_GRID:
        res = core.solve("cool", t_room, t_out,
                         OperatingPoint(float(n), 850.0, 850.0))
        if res.ok and res.q_capacity >= q_req:
            loud = res
            break
    _, quiet = opt.best("cool", t_room, t_out, q_req)
    print(_row("High-speed burst (typical AC behavior)", loud))
    print(_row("AURA-1 optimized continuous", quiet))
    print(f"  → {loud.dba_indoor - quiet.dba_indoor:.1f} dBA quieter indoors, "
          f"{(1 - quiet.w_total/loud.w_total) * 100:.0f}% less power")


def design_sweep() -> None:
    """Hardware design-space search (`--optimize`): sweep coil UA, fan size
    and compressor displacement; score each candidate on seasonal-weighted
    energy at standard rating points plus a noise penalty. The optimizer
    'optimizes the design', not just the operating point."""
    print("\n=== DESIGN SWEEP — optimizing the hardware itself ===")
    print("  scoring: seasonal watts + 0.3·dBA_max + packaging penalty")
    cases = [  # (label, ambient_C, mode, t_room_C, required W, weight)
        ("cool95", 35.0, "cool", 26.7, 3200.0, 0.30),
        ("cool82", 27.8, "cool", 26.7, 1500.0, 0.30),
        ("heat47",  8.3, "heat", 21.1, 3200.0, 0.22),
        ("heat17", -8.3, "heat", 21.1, 3200.0, 0.18)]
    base_ua = (900.0, 1400.0)
    base_fan = (7.3e-4, 1.6e-3)
    base_p = (38.0, 55.0)
    rows = []
    for ua_s in (0.80, 1.00, 1.25):
        for fan_s in (0.85, 1.00, 1.15):
            for vd in (10.0e-6, 12.0e-6, 14.0e-6):
                c = HeatPumpCore()
                c.coil_in.ua_nom = base_ua[0] * ua_s
                c.coil_out.ua_nom = base_ua[1] * ua_s
                for fan, k0, p0 in ((c.fan_in, base_fan[0], base_p[0]),
                                    (c.fan_out, base_fan[1], base_p[1])):
                    fan.k = k0 * fan_s
                    fan.p_nom = p0 * fan_s ** 2   # bigger fan, same aero eff.
                c.comp.V_D = vd
                o = QuietOptimizer(c)
                score = e_w = 0.0
                dba_max = 0.0
                ok_all = True
                for _lbl, tout, mode, troom, qreq, wgt in cases:
                    _, res = o.best_power(mode, K(troom), K(tout), qreq)
                    if not res.ok:
                        ok_all = False
                        break
                    e_w += res.w_total * wgt
                    dba_max = max(dba_max, res.dba_indoor)
                if not ok_all:
                    continue
                # packaging penalty: coil volume/mass/charge grow with UA,
                # and the drum only fits ~1.3× the baseline face area;
                # without this the sweep trivially picks infinite coil
                pack = 60.0 * max(0.0, ua_s - 1.0) + \
                    20.0 * max(0.0, fan_s - 1.0)
                if ua_s > 1.30:
                    continue                  # physically doesn't fit
                score = e_w + 0.3 * dba_max * 10.0 + pack
                rows.append((score, e_w, dba_max, ua_s, fan_s,
                             vd * 1e6))
    rows.sort()
    print("  rank  UA×    fan×   Vd(cm³)   W_seas   dBA_max")
    for i, (sc, e_w, dm, ua_s, fan_s, vd) in enumerate(rows[:8]):
        tag = "  <== current" if abs(ua_s - 1.0) < 1e-9 and \
            abs(fan_s - 1.0) < 1e-9 and abs(vd - 12.0) < 1e-9 else ""
        print(f"  {i+1:4d}  {ua_s:4.2f}  {fan_s:5.2f}  {vd:7.0f}   "
              f"{e_w:6.0f}   {dm:5.1f}{tag}")
    best = rows[0]
    print(f"  → best: UA×{best[3]:.2f}, fan×{best[4]:.2f}, "
          f"Vd={best[5]:.0f} cm³ — {best[1]:.0f} W seasonal-weighted, "
          f"{best[2]:.1f} dBA worst-case")


def export_blueprint(core: HeatPumpCore, opt: QuietOptimizer,
                     path: str = "AURA-1_blueprint.json") -> str:
    """Frozen digital design package — every parameter the model uses,
    plus the verified operating envelope and control law. This is the
    'as-designed' record a prototype build would be checked against."""
    import json
    env = {}
    for lbl, mode, tc in (("cool_35C", "cool", 35.0),
                          ("heat_8C", "heat", 8.3),
                          ("heat_-8C", "heat", -8.3),
                          ("heat_-15C", "heat", -15.0),
                          ("heat_-40C", "heat", -40.0),
                          ("heat_-50C", "heat", -50.0)):
        pt, res = opt.best(mode, K(21.1 if mode == "heat" else 26.7),
                           K(tc), 1500.0)
        env[lbl] = {"q_w": round(res.q_capacity), "w_w": round(res.w_total),
                    "cop": round(res.cop, 2), "dba_in": round(res.dba_indoor, 1),
                    "t_dis_c": round(C(res.t_discharge), 1),
                    "ok": res.ok, "evi": res.evi_active,
                    "notes": res.notes}
    bp = {
        "unit": "AURA-1 window-replacement heat pump",
        "actuation": core.actuator.method,
        "refrigerant": dict(REFRIGERANT_SPEC,
                            property_model="Peng-Robinson EOS, Wagner psat, "
                                           "Watson hfg"),
        "materials": MATERIALS,
        "electrical": ELECTRICAL_SPEC,
        "performance_spec": PERFORMANCE_SPEC,
        "controls_spec": CONTROLS_SPEC,
        "safety_spec": SAFETY_SPEC,
        "installation_spec": INSTALLATION_SPEC,
        "dimensional_spec": DIMENSIONAL_SPEC,
        "environmental_spec": ENVIRONMENT_SPEC,
        "component_detail": COMPONENT_DETAIL,
        "regulatory_spec": REGULATORY_SPEC,
        "service_spec": SERVICE_SPEC,
        "cost_spec": COST_SPEC,
        "geometry": {"drum_od_mm": 540, "adapter_frame_mm": [580, 580],
                     "rough_opening_mm": [711, 610],
                     "flip_axis": "horizontal, sill-parallel, EPDM seals"},
        "compressor": {"type": "twin-rotary inverter, EVI multi-port",
                       "v_d_cm3": core.comp.V_D * 1e6,
                       "n_range_rps": [core.comp.N_MIN, core.comp.N_MAX],
                       "eta_is_peak": core.comp.ETA_IS_PEAK,
                       "t_dis_max_c": C(core.t_dis_max)},
        "coils": {"indoor": {"ua_nom_w_k": core.coil_in.ua_nom,
                             "m_air_ref": core.coil_in.m_air_ref,
                             "dp_ref_pa": core.coil_in.dp_ref},
                  "outdoor": {"ua_nom_w_k": core.coil_out.ua_nom,
                              "m_air_ref": core.coil_out.m_air_ref,
                              "dp_ref_pa": core.coil_out.dp_ref},
                  "maldis_derate": MicroChannelCoil.MALDIS},
        "fans": {"indoor": {"kg_per_rpm": core.fan_in.k,
                            "rpm_nom": core.fan_in.rpm_nom,
                            "p_nom_w": core.fan_in.p_nom,
                            "l_nom_dba": core.fan_in.l_nom},
                 "outdoor": {"kg_per_rpm": core.fan_out.k,
                             "rpm_nom": core.fan_out.rpm_nom,
                             "p_nom_w": core.fan_out.p_nom,
                             "l_nom_dba": core.fan_out.l_nom}},
        "expansion": {"eev_subcool_k": ExpansionTrain.SUBCOOL_K,
                      "superheat_k": ExpansionTrain.SUPERHEAT_K,
                      "ihx_eff": ExpansionTrain.IHX_EFF,
                      "ejector_recovery": ExpansionTrain.EJECTOR_WORK_RECOVERY,
                      "evi_injection_base": ExpansionTrain.EVI_INJECTION,
                      "evi_staging_c": {"1_port": "<-3", "2_port": "<-18",
                                        "3_port": "<-30"},
                      "liquid_quench": "last-port liquid injection, "
                                       "T_dis clamp bisection"},
        "losses": {"dp_suc_pa_at_ref": core.dp_suc,
                   "dp_dis_pa_at_ref": core.dp_dis,
                   "m_dot_ref_kg_s": core.m_dot_ref,
                   "dp_scaling": "proportional to m_dot^2",
                   "shell_loss_frac": core.shell_frac,
                   "crankcase_w_below_-5C": core.w_crankcase,
                   "valve_loss_frac": ModeActuator.VALVE_LOSS,
                   "standby_w": core.w_standby,
                   "atomizer_w": core.w_atomizer},
        "aux": {"wind_ua_gain": core.wind_ua_gain,
                "solar_coil_model": "solar_w param raises evap air-side T",
                "evi_threshold_c": core.evi_threshold_c},
        "operating_envelope": env,
        "control_law": {
            "optimizer": "two-stage grid scan, min dBA then watts, "
                         "dba_cap for night mode",
            "cycling": "duty cycle + 8% restart loss below 1.6x overshoot",
            "defrost": "frost>0.55 triggers ~4 min reversed-cycle defrost",
            "night_cap_dba": 25.0},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bp, f, indent=2, ensure_ascii=False)
    return path


# Design-spec data — the physical bill-of-materials the numbers stand for.
# Not solved by the model (it is an engineering *spec*, not physics), but
# required for the blueprint to read as a buildable product.
REFRIGERANT_SPEC = {
    "name": "R-32 (difluoromethane, CH2F2)",
    "molar_mass_g_mol": 52.02,
    "gwp100": 675, "odp": 0.0,
    "safety_class": "A2L (mildly flammable)",
    "nbp_c": -51.65, "t_crit_c": 78.1, "p_crit_mpa": 5.782,
    "charge_g_est": 450,          # micro-channel, charge-minimized
    "oil": "POE (polyol ester), ~220 ml in compressor sump",
    "alternates_considered": {
        "R-290 (propane)": "GWP≈3 but A3 flammable — charge limit fights "
                           "a sealed rotating drum; tracked as future variant",
        "R-454B": "GWP 466, ~10% lower capacity than R-32, near drop-in",
        "R-410A": "GWP 2088 — rejected, phased down",
        "R-744 (CO2)": "transcritical cycle, ~10 MPa high side — "
                       "different machine entirely",
    },
}
MATERIALS = {
    "adapter_frame": "6063-T5 extruded aluminum, powder-coated, "
                     "thermal-break polymer insert",
    "drum_shell": "deep-drawn 5052-H32 aluminum, 1.5 mm wall, "
                  "closed-cell elastomeric liner at end caps",
    "drum_end_caps": "PA6 + 30% glass fiber, twin louvered apertures",
    "perimeter_seals": "dual-durometer EPDM rotating lip seals ×2",
    "bearing_axis": "303 stainless axle Ø12 mm; sealed stainless "
                    "deep-groove bearings (608-2RS class), -40 °C grease",
    "coils": "AA3103 aluminum micro-channel flat tube + louvered fin; "
             "indoor bank hydrophilic-coated (condensate drainage); "
             "both epoxy e-coated for corrosion",
    "compressor": "hermetic steel-shell twin-rotary, NdFeB IPM motor, "
                  "EVI multi-port shell; elastomeric isolation mount",
    "fan_indoor": "cross-flow wheel, glass-filled ABS",
    "fan_outdoor": "axial propeller, glass-filled PP",
    "refrigerant_lines": "copper ACR 3/8-in + 1/4-in, brazed joints; "
                         "suction line elastomeric-insulated",
    "expansion_module": "brass-body EEV, machined-brass ejector, "
                        "brazed-plate 316L stainless IHX, brass EVI "
                        "flash tank with 3 injection ports",
    "condensate_tray": "ABS, sloped to atomizer slinger ring",
    "louvers": "extruded aluminum, fixed 20° pitch, both end caps",
    "fasteners": "A2-70 stainless",
    "electronics": "conformal-coated inverter + controls board, "
                   "IP54 enclosure at frame lower rail",
}
ELECTRICAL_SPEC = {
    "supply": "120/230 V ~ 50/60 Hz universal inverter input",
    "max_input_w": 1100.0,        # comp ~850 + fans ~95 + aux + margin
    "standby_w": 2.0,
    "inrush": "negligible — inverter soft-start",
    "cord_plug": "NEMA 5-15P (120 V) / per-market cord for 230 V",
    "ip_rating": "electronics bay IP54; drum interior splash-guarded",
    "mass_kg_est": 34.0,          # drum core ~21 + frame/panels ~13
    "rated_current_a": 6.8,       # at rated cooling point on 120 V
    "max_current_a": 9.2,         # at max_input_w / 120 V
    "mca_a": 11.5,                # minimum circuit ampacity
    "breaker_a": 15,              # dedicated or shared 15 A branch
    "power_factor": ">=0.98 — active PFC front end on inverter",
    "voltage_tolerance": "±10 % (108–132 V / 207–253 V)",
    "cord_length_m": 1.8,
}
PERFORMANCE_SPEC = {
    "capacity_class": "12 kBTU/h class (nominal 3.5 kW cool / 3.4 kW heat)",
    "rated_cooling": "3539 W @ 35 °C amb / 26.7 °C room — COP 4.86",
    "rated_heating": "3432 W @ 8.3 °C amb / 21.1 °C room — COP 4.61",
    "modulation": "1.7–5.0 kW continuous (compressor 15–120 rps, "
                  "inverter); duty-cycled below ~1.7 kW",
    "airflow_indoor": "0.55 kg/s ≈ 980 CFM nominal @ 750 rpm "
                      "(oversized cross-flow; quiet duty runs far slower)",
    "airflow_outdoor": "0.96 kg/s ≈ 1720 CFM nominal @ 600 rpm (axial)",
    "dehumidification": "≈2.2 L/h latent removal at 26.7 °C/50 % RH "
                        "room, 35 °C ambient (model-derived)",
    "ambient_envelope": "cooling +10…+46 °C; heating −50…+24 °C "
                        "(EVI×3 + quench holds T_dis ≤112 °C at −50)",
    "sound": "indoor 17–36 dBA operating; 25 dBA night cap; "
             "compressor −10 dB isolation; outdoor <50 dBA",
    "seasonal": "SEER ≈22.4 · HSPF ≈12.6 (bin estimate)",
}
CONTROLS_SPEC = {
    "control_board": "conformal-coated inverter + MCU, IP54 bay",
    "sensors": "indoor/outdoor dry-bulb, coil-face thermistors, "
               "suction/discharge pressure transducers, line current, "
               "fan tach, frost proxy via coil-T + ambient-RH",
    "algorithms": "PI + feedforward capacity control, ±0.15 K deadband; "
                  "minimum-noise operating-point optimizer; duty cycling "
                  "below 1.6× overshoot (8 % restart penalty); adaptive "
                  "reverse-cycle defrost (metered, ≤4 min); EVI staging + "
                  "discharge-temperature quench control",
    "interfaces": "IR remote + local capacitive panel on indoor face; "
                  "Wi-Fi module footprint for app/demand-response "
                  "(OpenADR-ready API)",
    "modes": "cool / heat / auto / fan-only / night-quiet (25 dBA cap "
             "22:00–07:00) / dry-bias",
}
SAFETY_SPEC = {
    "flip_interlock": "drum rotation inhibited while powered; dual "
                      "locking pins + positional microswitch confirm "
                      "0°/180° detent before restart",
    "seal_monitor": "perimeter-seal compression switch; fault if "
                    "unseated >3 mm",
    "refrigerant": "A2L-safe charge 450 g (< room-flammability limit "
                   "for smallest install room); leak-detect sensor in "
                   "indoor bay with auto-shutoff + alarm",
    "pressure": "high-pressure cutout 4.2 MPa; relief per ISO 5149 "
                "sizing on liquid line",
    "thermal": "discharge-temperature limit 120 °C (quench-enforced), "
               "compressor motor thermal protector",
    "mechanical": "anti-tip mounting rails; child/pet louver-guard "
                  "spacing <8 mm; finger-safe fan shrouds",
    "electrical": "GFCI-compatible cord, fused board, surge clamp, "
                  "standby <2 W, soft-start (no inrush)",
}
INSTALLATION_SPEC = {
    "opening_fit": "double-hung / slider / casement rough openings "
                   "711×610 mm class via telescoping adapter frame "
                   "(custom panels for full sash replacement)",
    "weight": "≈34 kg installed — gas-strut-assisted pivot permits "
              "one-person seasonal flip",
    "daylight": "optional translucent polycarbonate side panels "
                "preserve ~60 % of glazed area around the core",
    "sealing": "dual-durometer EPDM perimeter seals + thermal-break "
               "insert → 6 W/K effective infiltration (vs 30–50 typical)",
    "service": "indoor-side-only filter access (washable MERV-8 "
               "electrostatic); sealed refrigerant circuit — no field "
               "charge required",
    "durability": "15–20 yr design life; e-coated coils, stainless "
                  "fasteners, UV-stabilized polymers",
}
DIMENSIONAL_SPEC = {
    "installed_envelope": "adapter frame face 711 W × 610 H mm fills "
                          "rough opening; drum OD 540 mm",
    "depths_mm": "indoor projection 120 mm; outdoor projection 260 mm; "
                 "total depth through wall ~380 mm + sill bracket",
    "clearances_mm": "indoor: 300 mm in front for filter swing + service; "
                     "outdoor: 600 mm discharge clearance preferred",
    "sill_slope_deg": "±2° mounting tolerance; factory shims compensate "
                      "up to 4° out-of-level",
    "opening_types": "double-hung / slider / casement via telescoping "
                     "adapter; full sash-replacement panels optional",
    "shipping": "carton 780 × 700 × 480 mm, ~40 kg gross; core ships "
                "pre-charged, frame hardware separate box",
}
REGULATORY_SPEC = {
    "safety": "UL 60335-2-40 Ed.5 (A2L refrigerant clauses) / "
              "IEC 60335-2-40 — certification path, not yet certified",
    "refrigerant_handling": "ISO 5149 / EN 378 charge limits "
                            "(450 g A2L) — within unventilated-room "
                            "limit for >4 m² floor area",
    "performance_rating": "AHRI 210/240 rating methods; DOE CEER test "
                          "procedure (10 CFR 430 subpart B)",
    "emissions": "FCC Part 15 Class B (digital device); "
                 "CISPR 14-1 conducted/radiated",
    "chemistry": "RoHS 3 / REACH compliant BOM; no PFAS coatings; "
                 "SNAP-listed R-32",
    "energy_label": "ENERGY STAR room AC criteria exceedance ~1.8× "
                    "(CEER 15 req → modeled ~26 seasonal-equivalent)",
}
SERVICE_SPEC = {
    "warranty_target": "2 yr parts / 5 yr compressor (design target)",
    "maintenance": "indoor filter rinse monthly in season; annual seal "
                   "inspection; condensate path self-cleaning via atomizer",
    "service_ports": "factory-sealed circuit; two Schrader process "
                     "stubs behind service cover for recovery only — "
                     "no field charging",
    "diagnostics": "onboard LED fault codes + app history; sensor-fail "
                   "modes degrade to safe fixed-speed fallback",
    "reliability": "design life 15–20 yr; bearing/seal wear items "
                   "rated >10 yr; flip mechanism rated >50 cycles "
                   "(seasonal use ~2/yr → 25 yr margin)",
}
ENVIRONMENT_SPEC = {
    "operating": "cooling +10…+46 °C; heating −50…+24 °C; "
                 "RH ≤90 % non-condensing indoors",
    "storage": "−30…+60 °C, RH ≤95 % non-condensing",
    "altitude": "≤2000 m full rating; derate ~1 %/300 m above "
                "(air-density correction already in model)",
    "weather": "outdoor face rated for driven rain, −50 °C cold-soak "
               "start, 40 m/s wind gust; louver shed geometry",
    "corrosion": "1000 h salt-spray on coils (epoxy e-coat); coastal "
                 "package optional",
}
COMPONENT_DETAIL = {
    "coil_geometry": "micro-channel: 220 mm face × 52 mm deep × 3 tube "
                     "rows; 20 FPI louvered fins 0.10 mm; multiport flat "
                     "tubes 1.0×16 mm, 24 ports/tube",
    "design_pressures": "high side 4.5 MPa working / 6.75 MPa proof; "
                        "low side 2.5 MPa working / 3.75 MPa proof "
                        "(ISO 14903 leak-tight class A)",
    "leak_rate": "<1 g/yr refrigerant loss target (all-brazed joints, "
                 "no flares in field service path)",
    "eev": "500-step pulse EEV, 0–2.4 mm needle travel, closed-loop on "
           "suction superheat (5 K) + subcool (6 K)",
    "compressor_shell": "steel shell OD 118 mm × 170 mm tall, 12 cm³ "
                        "twin-rotary, POE 220 ml, EVI ports ×3 + quench",
    "fan_motors": "ECM BLDC 24 V DC bus, class-F insulation, ball "
                  "bearings, IP42, tach feedback",
    "bearings_axis": "608-2RS sealed deep-groove, L10 >40 000 h; "
                     "Ø12 mm 303 axle, ~180 N·m flip torque, gas struts "
                     "2× 350 N assist",
    "insulation": "closed-cell NBR/PVC elastomeric liner 6 mm "
                  "(λ≈0.034 W/mK) inside drum end caps + frame "
                  "thermal-break insert",
    "filter": "washable MERV-8 electrostatic panel 400×180 mm, "
              "indoor-side access only",
    "finish": "frame powder-coat RAL 7016 graphite 60 µm; drum "
              "anodized; indoor face matte anti-glare",
}
COST_SPEC = {
    "bom_estimate_usd": 520,      # modeled bill-of-materials target
    "msrp_target_usd": 1299,      # premium-position vs $350 commodity AC
    "install_cost_usd": "DIY ~0 / pro ~150 (15 min adapter frame)",
    "annual_energy_usd": "≈$95 cooling + $140 heating at $0.16/kWh "
                         "mixed climate (bin-model estimate)",
    "payback": "~3 yr vs window AC + resistance heat at typical rates",
}

def spec_sheet(core: HeatPumpCore) -> None:
    """One-page engineering spec for the modeled machine."""
    c, f_in, f_out = core.comp, core.fan_in, core.fan_out
    print("\n=== AURA-1 ENGINEERING SPEC SHEET (as modeled) ===")
    rs = REFRIGERANT_SPEC
    print(f"  Refrigerant      {rs['name']}")
    print(f"                   GWP {rs['gwp100']}, ODP {rs['odp']}, "
          f"{rs['safety_class']}, charge ≈{rs['charge_g_est']} g, {rs['oil']}")
    print(f"                   NBP {rs['nbp_c']} °C, Tc {rs['t_crit_c']} °C, "
          f"Pc {rs['p_crit_mpa']} MPa")
    print(f"  Compressor       twin-rotary inverter, {c.V_D*1e6:.0f} cm^3, "
          f"{c.N_MIN:.0f}–{c.N_MAX:.0f} rps, EVI multi-port + liquid quench")
    print(f"  Indoor coil      micro-channel, UA≈{core.coil_in.ua_nom:.0f} W/K "
          f"@ {core.coil_in.m_air_ref:.2f} kg/s")
    print(f"  Outdoor coil     micro-channel (flip-to-vertical oversized), "
          f"UA≈{core.coil_out.ua_nom:.0f} W/K @ {core.coil_out.m_air_ref:.2f} kg/s")
    print(f"  Indoor fan       ECM {f_in.airflow(f_in.rpm_nom):.2f} kg/s "
          f"@ {f_in.rpm_nom:.0f} rpm, {f_in.p_nom:.0f} W")
    print(f"  Outdoor fan      ECM {f_out.airflow(f_out.rpm_nom):.2f} kg/s "
          f"@ {f_out.rpm_nom:.0f} rpm, {f_out.p_nom:.0f} W")
    print("  Expansion        EEV + ejector recovery + suction-line IHX")
    print(f"  Mode actuation   {core.actuator.method} "
          f"({'no valve Δp' if core.actuator.method == 'flip' else '4-way valve'})")
    print(f"  EVI threshold    outdoor < {core.evi_threshold_c:.0f} °C (heating)")
    print(f"  Aux power        standby {core.w_standby:.0f} W + "
          f"condensate atomizer {core.w_atomizer:.0f} W (zero-drip)")
    print("  Acoustics        compressor −10 dB chassis isolation; optimizer "
          "targets minimum feasible dBA; night cap supported")
    print("  Envelope         airtight adapter frame, +6 W/K effective leak "
          "(typical window AC ≈ 30–50 W/K)")
    e = ELECTRICAL_SPEC
    print("  --- electrical/mechanical ---")
    print(f"  Supply           {e['supply']}")
    print(f"  Max input        {e['max_input_w']:.0f} W "
          f"(inrush: {e['inrush']})")
    print(f"  Mass (est.)      {e['mass_kg_est']:.0f} kg installed "
          f"| {e['ip_rating']}")
    print("  --- materials (bill of construction) ---")
    for k, v in MATERIALS.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- performance ---")
    for k, v in PERFORMANCE_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- controls ---")
    for k, v in CONTROLS_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- safety ---")
    for k, v in SAFETY_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- installation/service ---")
    for k, v in INSTALLATION_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- dimensions ---")
    for k, v in DIMENSIONAL_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- environmental limits ---")
    for k, v in ENVIRONMENT_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- regulatory (targets) ---")
    for k, v in REGULATORY_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- component detail ---")
    for k, v in COMPONENT_DETAIL.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- service & warranty ---")
    for k, v in SERVICE_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")
    print("  --- cost targets ---")
    for k, v in COST_SPEC.items():
        print(f"  {k.replace('_',' '):<16} {v}")


# =============================================================================
# §12b  VALIDATION — the "does it work in real life" harness
# =============================================================================
# Three independent proofs:
#   1. Conservation audit — Q_cond − Q_evap must equal refrigerant-side work
#      (m·w_act) to <1 % at every operating point. This catches bookkeeping
#      bugs exactly like the IHX one the audit was written for.
#   2. Carnot bound — no legal heat pump may exceed Carnot COP. Swept over
#      the full operating envelope; any violation = model bug.
#   3. Reference-property check — R-32 saturation pressure vs. published
#      values (NBP −51.65 °C, plus engineering-reference points, ±8 %).

R32_PSAT_REF = {   # °C → MPa, NIST-consistent engineering reference values
    -51.65: 0.1013, -40.0: 0.175, -20.0: 0.399,
    0.0: 0.804, 25.0: 1.69, 40.0: 2.45,
}


def validate(core: HeatPumpCore, opt: QuietOptimizer) -> None:
    r = RefrigerantR32
    print("\n=== VALIDATION ===")
    # -- 3. property check --------------------------------------------------
    print("  [3] R-32 saturation pressure vs reference:")
    worst = 0.0
    for tc, p_ref in R32_PSAT_REF.items():
        p_mod = r.psat(K(tc)) / 1e6
        err = abs(p_mod / p_ref - 1.0) * 100.0
        worst = max(worst, err)
        flag = "OK" if err < 8.0 else "FAIL"
        print(f"      T={tc:7.2f}°C  model={p_mod:7.4f} MPa  "
              f"ref={p_ref:7.4f} MPa  err={err:5.2f}%  {flag}")
    # -- 1. conservation audit ----------------------------------------------
    print("  [1] Energy-conservation audit "
          "(|Q_cond−Q_evap−W_thermo| / Q_evap < 1%):")
    audit_pts = [
        ("cool", K(26.0), K(35.0), OperatingPoint(60, 500, 500)),
        ("cool", K(26.0), K(45.0), OperatingPoint(90, 700, 800)),
        ("heat", K(21.0), K(8.0), OperatingPoint(80, 600, 500)),
        ("heat", K(21.0), K(-15.0), OperatingPoint(110, 700, 600)),  # EVI
        ("heat", K(21.0), K(-45.0), OperatingPoint(95, 700, 600)),   # EVI×3+quench
    ]
    worst_a = 0.0
    for mode, tr, to, pt in audit_pts:
        res = core.solve(mode, tr, to, pt)
        # conservation: Q_evap + W_thermo = Q_cond + Q_shell (shell loss
        # leaves the loop to ambient), whichever side is which
        q_evap = res.q_capacity if mode == "cool" else res.q_reject
        q_cond = res.q_reject if mode == "cool" else res.q_capacity
        delta = abs(q_evap + res.w_thermo - q_cond - res.q_shell_w)
        rel = delta / max(res.q_capacity, 1.0) * 100.0
        worst_a = max(worst_a, rel)
        flag = "OK" if rel < 1.0 else "FAIL"
        print(f"      {mode} out={C(to):6.1f}°C N={pt.n_rps:4.0f}  "
              f"imbalance={rel:6.3f}%  {flag}")
    # -- 2. Carnot bound sweep ----------------------------------------------
    print("  [2] Carnot-bound sweep (COP must be < Carnot everywhere):")
    viol = 0
    for mode in ("cool", "heat"):
        for to_c in (-20, -5, 10, 25, 40, 46):
            for n in (30, 70, 115):
                res = core.solve(mode, K(24 if mode == "cool" else 21),
                                 K(to_c), OperatingPoint(n, 600, 600))
                if res.ok and res.cop >= res.carnot_cop:
                    viol += 1
                    print(f"      VIOLATION {mode} out={to_c}°C N={n} "
                          f"COP={res.cop:.2f} ≥ Carnot={res.carnot_cop:.2f}")
    print(f"      {viol} violations across 36 swept points")
    verdict = (worst < 8.0 and worst_a < 1.0 and viol == 0)
    print(f"  VERDICT: {'PASS' if verdict else 'FAIL'}  "
          f"(psat worst {worst:.1f}%, audit worst {worst_a:.3f}%, "
          f"carnot violations {viol})")


# =============================================================================
# §12c MODEL DRAWING — to-scale engineering sheet, pyglet viewer + SVG export
# =============================================================================
# One geometry spec, two renderers. The machine is a Ø480 mm rotating drum
# inside a window-filling adapter frame; the drum flips 180° about a
# horizontal sealed-bearing axis to swap which coil faces the room.
#   --view  interactive pyglet window: animated 180° drum flip, refrigerant
#           particles colored by state, spinning ECM fans, live telemetry.
#   --svg   static to-scale sheet: side section + front elevation +
#           summer/winter flip states, dims, leaders, telemetry snapshot.
# Grey background + high-contrast bright inks throughout, per spec.

BG_GREY = (62, 63, 68)
INK = (235, 236, 240)          # primary label ink
INK_DIM = (168, 170, 176)      # secondary ink
ENVELOPE = (90, 92, 98)        # ghost chassis
HOT = (255, 120, 70)
COLD = (90, 170, 255)
MM = 0.5                       # px per mm — drawing is to scale

# ---------------------------------------------------------------------------
# TO-SCALE GEOMETRY — the real machine (not a generic schematic).
# AURA-1 is a drum inside a window-filling adapter frame. The drum rotates
# 180° about a horizontal, sill-parallel sealed-bearing axis: the coil that
# faces indoors in summer (evaporator) faces outdoors in winter after the
# flip, and vice-versa. Side-section view is the drawing's centerpiece;
# front elevation + summer/winter flip panels complete the sheet.
# All dims in px at MM px/mm; canvas 1440×900, y measured bottom-up.
# ---------------------------------------------------------------------------
DRUM = dict(cx=380, cy=470, r=135)         # Ø540 mm rotating core
FRAME = dict(x=235, y=325, w=290, h=290)   # adapter frame: 580×580 mm sec
WALL = dict(x=330, y=210, w=100, h=590)    # wall/opening band, ~200 mm
COIL_A = dict(x=258, y=415, w=26, h=110)   # outdoor-face coil 52×220 mm
COIL_B = dict(x=476, y=415, w=26, h=110)   # indoor-face coil
COMP = dict(x=350, y=368, w=60, h=64)      # compressor 120×128 mm
EXP = dict(x=355, y=380, w=60, h=24)       # EEV + ejector + IHX module
FAN_A = dict(cx=300, cy=515, r=25)         # outdoor axial fan Ø100 mm
FAN_B = dict(cx=460, cy=515, r=24)         # indoor cross-flow Ø96 mm
TRAY = dict(x=415, y=355, w=35, h=10)      # condensate tray + atomizer

FLOW_COLORS = {
    "discharge": (255, 92, 70), "condenser": (255, 160, 60),
    "liquid": (255, 205, 90), "expansion": (175, 120, 255),
    "evap_in": (120, 160, 255), "evaporator": (90, 140, 255),
    "suction": (90, 220, 230),
}


def _rot180(x, y):
    """Rotate a point 180° about the drum axis (the physical flip)."""
    return (2 * DRUM["cx"] - x, 2 * DRUM["cy"] - y)


def loop_segments(mode: str = "cool", flip: bool = False):
    """Refrigerant path inside the drum, in travel order.

    Segment tags describe what the refrigerant is doing, which is what the
    coil physically does — so for the FLIP machine the geometry rotates and
    the tags stay put (the same coil keeps its role, it just moves sides).
    For the VALVE machine geometry is fixed and roles swap via reversal."""
    base = [
        (365, 432, 365, 540, "discharge"), (365, 540, 271, 540, "discharge"),
        (271, 540, 271, 415, "condenser"), (271, 415, 271, 392, "liquid"),
        (271, 392, 489, 392, "liquid"),           # through expansion box
        (489, 392, 489, 415, "evap_in"),
        (489, 415, 489, 525, "evaporator"),
        (489, 525, 489, 545, "suction"), (489, 545, 395, 545, "suction"),
        (395, 545, 395, 432, "suction"),
    ]
    if flip:
        out = []
        for x1, y1, x2, y2, t in base:
            a = _rot180(x1, y1)
            b = _rot180(x2, y2)
            out.append((a[0], a[1], b[0], b[1], t))
        return out          # flip machine: same roles, rotated plumbing
    if mode == "cool":
        return base
    # valve reversal: reversed travel + coil roles swapped
    rev = [(x2, y2, x1, y1, t) for (x1, y1, x2, y2, t) in reversed(base)]
    swap = {"condenser": "evaporator", "evaporator": "condenser",
            "evap_in": "liquid", "liquid": "liquid",
            "discharge": "discharge", "expansion": "expansion",
            "suction": "suction"}
    return [(a, b, c, d, swap[t]) for (a, b, c, d, t) in rev]


def _path_table(mode: str = "cool", flip: bool = False):
    """Flatten segments → cumulative arc-length table for particle travel."""
    segs = loop_segments(mode, flip)
    pts, total = [], 0.0
    for x1, y1, x2, y2, tag in segs:
        ln = math.hypot(x2 - x1, y2 - y1)
        pts.append((x1, y1, x2, y2, ln, total, tag))
        total += ln
    return pts, total


# label leader-lines for the side section: (text, label_xy, target_xy)
LABELS_SIDE = [
    ("WALL / WINDOW OPENING", (150, 812), (382, 793)),
    ("AIR-TIGHT ADAPTER FRAME\n+ EPDM perimeter seals", (58, 655), (236, 555)),
    ("COIL A — summer condenser\n(outdoor face, oversized)", (60, 330), (260, 455)),
    ("COIL B — summer evaporator\n(indoor face)", (570, 340), (500, 455)),
    ("TWIN-ROTARY INVERTER\nCOMPRESSOR · 12 cm³ · EVI", (90, 200), (350, 370)),
    ("EXPANSION TRAIN\nEEV + ejector + IHX", (510, 290), (414, 390)),
    ("SEALED BEARING AXIS\ncore rotates 180°", (590, 700), (383, 473)),
    ("CONDENSATE TRAY +\nATOMIZER — zero drip", (600, 185), (449, 356)),
    ("axial fan\n(outdoor)", (115, 715), (300, 540)),
    ("cross-flow fan\n(indoor)", (470, 690), (462, 540)),
]


def export_svg(path: str, res: CycleResult = None) -> str:
    """Static to-scale engineering sheet. Grey bg, dims, leaders, labels."""
    W, H = 1440, 900
    s = ['<svg xmlns="http://www.w3.org/2000/svg" '
         f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         'font-family="Consolas, monospace">',
         '<defs><marker id="ah" markerWidth="8" markerHeight="8" '
         'refX="6" refY="3" orient="auto">'
         f'<path d="M0,0 L6,3 L0,6 z" fill="rgb{INK}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="rgb{BG_GREY}"/>']

    def yf(y):
        return H - y

    def ln(x1, y1, x2, y2, col=INK_DIM, w=1, dash=None, cap="butt"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<line x1="{x1:.0f}" y1="{yf(y1):.0f}" x2="{x2:.0f}" '
                 f'y2="{yf(y2):.0f}" stroke="rgb{col}" stroke-width="{w}"'
                 f' stroke-linecap="{cap}"{d}/>')

    def txt(x, y, t, size=12, fill=INK, anchor="start", weight="normal"):
        for i, line in enumerate(t.split("\n")):
            s.append(f'<text x="{x:.0f}" y="{yf(y):.0f}" dy="{i * 14}" '
                     f'font-size="{size}" fill="rgb{fill}" '
                     f'text-anchor="{anchor}" font-weight="{weight}">'
                     f'{line}</text>')

    def rect(d, fill, stroke=INK, dash=None, alpha=0.9, wgt=1.5):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<rect x="{d["x"]:.0f}" y="{yf(d["y"] + d["h"]):.0f}" '
                 f'width="{d["w"]:.0f}" height="{d["h"]:.0f}" '
                 f'fill="rgba{fill + (alpha,)}" stroke="rgb{stroke}" '
                 f'stroke-width="{wgt}"{dd}/>')

    def circ(cx, cy, r, fill=(74, 76, 84), stroke=INK, wgt=1.5, dash=None):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<circle cx="{cx:.0f}" cy="{yf(cy):.0f}" r="{r:.0f}" '
                 f'fill="rgba{fill + (0.9,)}" stroke="rgb{stroke}" '
                 f'stroke-width="{wgt}"{dd}/>')

    def arrow(x1, y1, x2, y2, col, w=3):
        ln(x1, y1, x2, y2, col, w, cap="round")
        a = math.atan2(y2 - y1, x2 - x1)
        for da in (2.6, -2.6):
            s.append(f'<line x1="{x2:.0f}" y1="{yf(y2):.0f}" '
                     f'x2="{x2 - 12 * math.cos(a + da):.0f}" '
                     f'y2="{yf(y2 - 12 * math.sin(a + da)):.0f}" '
                     f'stroke="rgb{col}" stroke-width="{w}" '
                     f'stroke-linecap="round"/>')

    def dim_h(x1, x2, y, label):   # horizontal dimension line
        ln(x1, y, x2, y, INK_DIM, 1)
        ln(x1, y - 6, x1, y + 6, INK_DIM, 1)
        ln(x2, y - 6, x2, y + 6, INK_DIM, 1)
        txt((x1 + x2) / 2, y + 10, label, 10, INK_DIM, anchor="middle")

    def dim_v(x, y1, y2, label):   # vertical dimension line
        ln(x, y1, x, y2, INK_DIM, 1)
        ln(x - 6, y1, x + 6, y1, INK_DIM, 1)
        ln(x - 6, y2, x + 6, y2, INK_DIM, 1)
        txt(x + 8, (y1 + y2) / 2 + 4, label, 10, INK_DIM)

    # ===================== title block =====================
    txt(40, 858, "AURA-1 — WINDOW-REPLACEMENT HEAT PUMP", 21, INK,
        weight="bold")
    txt(40, 836, "rotating-core 180° flip · R-32 · inverter + ejector + "
        "EVI + IHX · all views to scale (0.5 px/mm)", 11, INK_DIM)

    # ===================== PANEL A — side section =====================
    txt(80, 812, "SIDE SECTION — SUMMER (cooling)", 13, INK_DIM)
    rect(WALL, (50, 52, 57), INK_DIM)                    # wall band
    for yy in range(215, 785, 22):                       # wall hatch
        ln(WALL["x"] + 4, yy, WALL["x"] + 30, yy + 14, (72, 75, 82), 1)
    txt(382, 802, "WALL", 9, INK_DIM, anchor="middle")
    rect(FRAME, (70, 72, 78), INK, wgt=2)                # adapter frame
    # perimeter seals where drum meets frame
    for sx, sy in ((380, 610), (380, 330)):
        rect(dict(x=sx - 18, y=sy - 4, w=36, h=8), (140, 120, 60), (0, 0, 0))

    circ(DRUM["cx"], DRUM["cy"], DRUM["r"], (66, 68, 74), INK, 2.5)
    circ(DRUM["cx"], DRUM["cy"], 6, (200, 200, 205), INK, 1.5)  # axis cap
    # coils with fin striping
    for coil in (COIL_A, COIL_B):
        rect(coil, (78, 82, 92), INK)
        for i in range(6):
            xx = coil["x"] + 6 + i * 4
            ln(xx, coil["y"] + 6, xx, coil["y"] + coil["h"] - 6,
               (110, 116, 128), 1)
    rect(COMP, (88, 96, 90), INK, wgt=2)                 # compressor
    rect(EXP, (96, 80, 100), INK)                        # expansion train
    rect(TRAY, (70, 90, 100), INK_DIM)                   # condensate tray
    for f in (FAN_A, FAN_B):                             # fans + blades
        circ(f["cx"], f["cy"], f["r"], (60, 66, 74), INK_DIM)
        for k in range(3):
            a = math.radians(k * 120 + 30)
            ln(f["cx"], f["cy"], f["cx"] + f["r"] * 0.8 * math.cos(a),
               f["cy"] + f["r"] * 0.8 * math.sin(a), INK_DIM, 3)

    # refrigerant loop (summer orientation)
    for x1, y1, x2, y2, tag in loop_segments("cool"):
        ln(x1, y1, x2, y2, FLOW_COLORS[tag], 4, cap="round")

    # airflow arrows
    for yy in (435, 470, 505):
        arrow(120, yy, 252, yy, (120, 150, 180))         # outdoor intake
        arrow(650, yy, 508, yy, (150, 160, 170))         # indoor return
    arrow(285, 548, 212, 622, HOT)                       # hot exhaust out
    arrow(482, 543, 572, 618, COLD)                      # cool supply in
    txt(96, 555, "outdoor air", 10, (120, 150, 180))
    txt(605, 555, "room air", 10, (150, 160, 170), anchor="end")

    # 180° flip arc around the drum
    cx, cy, ra = DRUM["cx"], DRUM["cy"], DRUM["r"] + 42
    x1, y1 = cx + ra * math.cos(math.radians(25)), cy + ra * math.sin(math.radians(25))
    x2, y2 = cx + ra * math.cos(math.radians(205)), cy + ra * math.sin(math.radians(205))
    s.append(f'<path d="M {x1:.0f} {yf(y1):.0f} A {ra} {ra} 0 0 0 '
             f'{x2:.0f} {yf(y2):.0f}" fill="none" '
             f'stroke="rgb{FLOW_COLORS["expansion"]}" stroke-width="2.5" '
             'stroke-dasharray="10 7" marker-end="url(#ah)"/>')
    txt(cx + ra + 14, cy + 120, "CORE ROTATES 180°\n→ winter mode", 11,
        FLOW_COLORS["expansion"])

    # labels with leader lines
    for label, (lx, ly), (tx, ty) in LABELS_SIDE:
        ln(lx + 4, ly - 4, tx, ty, INK_DIM, 1, dash="3 3")
        txt(lx, ly, label, 10, INK)

    # dimension callouts (to scale)
    dim_h(DRUM["cx"] - DRUM["r"], DRUM["cx"] + DRUM["r"], 300,
          "Ø 540 mm rotating core")
    dim_v(560, FRAME["y"], FRAME["y"] + FRAME["h"], "580 mm")
    dim_h(FRAME["x"], FRAME["x"] + FRAME["w"], 645, "580 mm unit depth")
    # scale bar
    ln(70, 92, 170, 92, INK, 2)
    for i, mm in enumerate((0, 50, 100)):
        ln(70 + i * 50, 87, 70 + i * 50, 97, INK, 2)
        txt(70 + i * 50, 104, str(mm), 9, INK_DIM, anchor="middle")
    txt(175, 95, "mm", 9, INK_DIM)
    txt(95, 230, "OUTDOORS", 13, INK_DIM)
    txt(560, 230, "INDOORS", 13, INK_DIM)

    # refrigerant state legend
    lx, ly = 70, 150
    txt(lx, ly + 20, "REFRIGERANT STATES", 10, INK_DIM)
    for i, (tag, col) in enumerate(FLOW_COLORS.items()):
        xx = lx + (i % 4) * 130
        yy = ly - (i // 4) * 18
        s.append(f'<rect x="{xx}" y="{yf(yy) - 9}" width="14" height="7" '
                 f'fill="rgb{col}"/>')
        txt(xx + 20, yy, tag, 9, INK)

    # ===================== PANEL B — front elevation =====================
    txt(890, 832, "FRONT ELEVATION — indoor face (summer)", 13, INK_DIM)
    rect(dict(x=890, y=500, w=355, h=305), (56, 58, 63), INK, wgt=2)
    rect(dict(x=905, y=517, w=325, h=270), (70, 72, 78), INK)
    for i in range(7):                                   # grille louvers
        yy = 545 + i * 32
        ln(915, yy, 1220, yy, (95, 98, 105), 5)
    ln(905, 652, 1230, 652, INK_DIM, 1, dash="8 6")       # flip seam
    for bx in (890, 1245):                               # bearing caps
        circ(bx, 652, 11, (90, 94, 102), INK)
    dim_h(890, 1245, 482, "711 mm opening")
    dim_v(872, 500, 805, "610 mm")
    txt(1067, 508, "540 mm core face", 10, INK_DIM, anchor="middle")

    # ===================== PANEL C — seasonal flip =====================
    txt(890, 455, "SEASONAL REVERSAL — the core physically rotates 180°",
        13, INK_DIM)

    def mini_unit(cx, cy, r, winter):
        s.append(f'<rect x="{cx - 9}" y="{yf(cy + 105)}" width="18" '
                 f'height="210" fill="rgb(50,52,57)"/>')
        circ(cx, cy, r, (66, 68, 74), INK, 2)
        lx_c, rx_c = (cx - r + 28), (cx + r - 28)
        lc = COLD if winter else HOT        # outdoor-side coil colour
        rc_ = HOT if winter else COLD       # indoor-side coil colour
        rect(dict(x=lx_c - 12, y=cy - 40, w=24, h=80), lc, INK_DIM)
        rect(dict(x=rx_c - 12, y=cy - 40, w=24, h=80), rc_, INK_DIM)
        circ(cx, cy + 20, 10, (88, 96, 90), INK)         # compressor dot
        # room-side supply arrow (right = indoors)
        arrow(cx + r + 30, cy, cx + r + 95, cy, COLD if not winter else HOT)
        circ(cx, cy, 4, (200, 200, 205), INK)            # axis cap

    mini_unit(960, 320, 62, winter=False)
    txt(960, 215, "SUMMER\nevaporator indoors\n→ room cooled", 10, INK,
        anchor="middle")
    mini_unit(1230, 320, 62, winter=True)
    txt(1230, 215, "WINTER — flipped 180°\ncondenser indoors\n→ room heated",
        10, INK, anchor="middle")
    # big flip arrow between minis
    s.append(f'<path d="M 1020 {yf(360)} A 95 95 0 0 1 1145 {yf(360)}" '
             f'fill="none" stroke="rgb{FLOW_COLORS["expansion"]}" '
             'stroke-width="3" marker-end="url(#ah)"/>')
    txt(1083, 395, "FLIP", 12, FLOW_COLORS["expansion"], anchor="middle")

    # ===================== telemetry strip =====================
    if res is not None:
        txt(760, 155, "STATE @ rating point", 10, INK_DIM)
        lines = [
            f"mode {res.mode} ({res.actuation})   ",
            f"Te {C(res.t_evap):5.1f}°C   Tc {C(res.t_cond):5.1f}°C   "
            f"T2 {C(res.t_discharge):5.1f}°C",
            f"p {res.p_low/1e6:.2f}/{res.p_high/1e6:.2f} MPa   "
            f"ṁ {res.m_dot_ref*1000:.0f} g/s",
            f"Q {res.q_capacity:.0f} W   W {res.w_total:.0f} W   "
            f"COP {res.cop:.2f} ({100*res.carnot_fraction:.0f}% Carnot)",
            f"{res.dba_indoor:.0f} dBA in / {res.dba_outdoor:.0f} dBA out   "
            f"EVI {'ON' if res.evi_active else 'off'}",
        ]
        for i, ln_ in enumerate(lines):
            txt(760, 132 - i * 20, ln_, 11, INK)

    s.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s))
    return path


TAB_NAMES = ("MODEL", "CYCLE", "SPECS", "VALIDATION", "3D")


def run_viewer(core: HeatPumpCore, opt: QuietOptimizer) -> None:
    """Interactive tabbed model window (pyglet), reference-code style.
    Tab bar across the top — click a tab or press 1–5 / TAB:
      1 MODEL       to-scale side section; live refrigerant + 180° flip
      2 CYCLE       live log-p/h diagram + state-point table
      3 SPECS       engineering spec sheet
      4 VALIDATION  conservation audit + Carnot sweep + property check
      5 3D          lit OpenGL assembly — orbit, zoom, exploded view
    Model keys: ←/→ compressor · ↑/↓ fans · c/h mode · f flip/valve ·
    o quiet-optimize · space pause · F11 fullscreen · esc quit.
    3D keys: drag orbit · wheel zoom · x explode.
    Window is resizable + launches fullscreen; 2D content letterboxes."""
    import pyglet
    from pyglet import shapes, text
    from pyglet.window import key as _key
    from pyglet.window import mouse as _mouse

    W, Hh = 1440, 900
    # Request a compatibility context: the driver returns 4.6-compat, so
    # fixed-function calls (3D page) and pyglet shaders (2D pages) both
    # work. double_buffer=True is explicit — without it the driver may
    # hand back a single-buffered format, and then glDrawBuffer(GL_BACK)
    # fails and immediate-mode geometry is never presented.
    _cfg = pyglet.gl.Config(major_version=2, minor_version=1,
                            double_buffer=True, depth_size=24)
    win = pyglet.window.Window(W, Hh, caption="AURA-1 — model (live)",
                               config=_cfg, resizable=True)
    win.set_fullscreen(True)       # launch fullscreen; F11 toggles
    chrome = pyglet.graphics.Batch()       # always drawn: title, tab bar
    bg_b = pyglet.graphics.Batch()         # fullscreen grey bg (2D pages)
    static = pyglet.graphics.Batch()       # MODEL page statics
    batch = pyglet.graphics.Batch()        # MODEL page dynamics
    cyc_b = pyglet.graphics.Batch()        # CYCLE page
    spe_b = pyglet.graphics.Batch()        # SPECS page
    val_b = pyglet.graphics.Batch()        # VALIDATION page
    leg_b = pyglet.graphics.Batch()        # 3D page legend + float labels
    pages = [static, cyc_b, spe_b, val_b]
    state = {"pt": OperatingPoint(60.0, 500.0, 500.0), "res": None,
             "mode": "cool", "paused": False, "accum": 0.0,
             "ang": 0.0, "ang_t": 0.0, "fa_a": 0.0, "fb_a": 0.0,
             "primed": False, "tab": 0, "rx": -22.0, "ry": -32.0,
             "zoom": 1500.0, "expl": 0.0, "expl_t": 0.0}
    dcx, dcy = DRUM["cx"], DRUM["cy"]

    # pyglet 2.1.x quirk: GL buffers created before the context is current
    # silently drop the initial vertex upload. Every created object is
    # collected and self-assigned (.x = .x) once inside the event loop to
    # force a re-upload; otherwise untouched static geometry never renders.
    primed = []

    class _S:
        def __getattr__(self, name):
            f = getattr(shapes, name)
            def mk(*a, **k):
                o = f(*a, **k)
                primed.append(o)
                return o
            return mk

    def L(*a, **k):
        o = text.Label(*a, **k)
        primed.append(o)
        return o

    S = _S()

    # Letterbox scaling: all 2D content is authored in a fixed 1440×900
    # model space; the window may be any size (resizable/fullscreen), so
    # a view matrix scales and centers it preserving aspect. Mouse and
    # projected-label coordinates take the inverse transform.
    def _xform():
        w_, h_ = win.width, win.height
        sc = min(w_ / W, h_ / Hh)
        return sc, (w_ - W * sc) / 2.0, (h_ - Hh * sc) / 2.0

    def _apply_view():
        sc, ox, oy = _xform()
        from pyglet.math import Mat4, Vec3
        win.view = Mat4.from_translation(Vec3(ox, oy, 0.0)) @ \
            Mat4.from_scale(Vec3(sc, sc, 1.0))

    def to_model(x, y):
        sc, ox, oy = _xform()
        return (x - ox) / sc, (y - oy) / sc

    def rot(cx, cy, x, y, th):
        dx, dy = x - cx, y - cy
        return cx + dx * math.cos(th) - dy * math.sin(th), \
            cy + dx * math.sin(th) + dy * math.cos(th)

    def arrow(x1, y1, x2, y2, col, w=3, bt=None):
        bt = static if bt is None else bt
        parts = [S.Line(x1, y1, x2, y2, w, col, batch=bt)]
        a = math.atan2(y2 - y1, x2 - x1)
        for da in (2.6, -2.6):
            parts.append(S.Line(x2, y2, x2 - 11 * math.cos(a + da),
                                y2 - 11 * math.sin(a + da), w, col,
                                batch=bt))
        return parts

    # ================= CHROME (always drawn) =================
    S.Rectangle(0, 0, W, Hh, BG_GREY, batch=bg_b)
    L("AURA-1", x=24, y=860, font_size=19, weight="bold",
      color=INK + (255,), batch=chrome)
    L("window-replacement reversible heat pump — live physics model",
      x=140, y=866, font_size=10, color=INK_DIM + (255,), batch=chrome)
    tab_ui = []   # (rect, label, x0, x1)
    for i, name in enumerate(TAB_NAMES):
        bx = 520 + i * 168
        S.Rectangle(bx, 852, 156, 34, (50, 52, 58), batch=chrome)
        rr = S.Rectangle(bx, 852, 156, 4, FLOW_COLORS["expansion"],
                         batch=chrome)
        lb = L(f"{i + 1}  {name}", x=bx + 78, y=862, font_size=11,
               anchor_x="center", color=INK_DIM + (255,), batch=chrome)
        tab_ui.append((rr, lb, bx, bx + 156))
    L("TAB / 1-5 switch pages · MODEL: ←/→ comp · ↑/↓ fans · c/h mode · "
      "f flip/valve · o optimize · space · 3D: drag orbit, wheel zoom, "
      "x explode",
      x=24, y=10, font_size=9, color=INK_DIM + (255,), batch=chrome)

    def set_tab(i):
        state["tab"] = i
        for k, (rr, lb, _, _x1) in enumerate(tab_ui):
            on = k == i
            rr.color = FLOW_COLORS["expansion"] if on else (58, 60, 66)
            rr.height = 4 if on else 2
            lb.color = (INK if on else INK_DIM) + (255,)

    # ================= PAGE 1 — MODEL (to-scale side section) =================
    L("rotating-core 180° flip · 0.5 px/mm · side section",
      x=40, y=830, font_size=10, color=INK_DIM + (255,), batch=static)
    S.Rectangle(WALL["x"], WALL["y"], WALL["w"], WALL["h"],
                color=(50, 52, 57), batch=static)
    for yy in range(215, 785, 22):                      # wall hatch
        S.Line(WALL["x"] + 4, yy, WALL["x"] + 30, yy + 14, 1,
               (72, 75, 82), batch=static)
    S.BorderedRectangle(FRAME["x"], FRAME["y"], FRAME["w"],
                        FRAME["h"], border=2, color=(0, 0, 0, 0),
                        border_color=INK, batch=static)
    for sy in (610, 330):                               # perimeter seals
        S.Rectangle(dcx - 18, sy - 4, 36, 8, (140, 120, 60), batch=static)
    S.Arc(dcx, dcy, DRUM["r"], thickness=3, color=INK, batch=static)
    rim_dot = S.Circle(dcx + DRUM["r"], dcy, 6,
                       color=FLOW_COLORS["expansion"], batch=batch)
    # airflow arrows
    for yy in (435, 470, 505):
        arrow(120, yy, 252, yy, (120, 150, 180))        # outdoor intake
        arrow(650, yy, 508, yy, (150, 160, 170))        # indoor return
    exhaust_arrow = arrow(285, 548, 212, 622, HOT)      # outdoor exhaust
    supply_arrow = arrow(482, 543, 572, 618, COLD)      # indoor supply
    L("OUTDOORS", x=95, y=228, font_size=11,
      color=INK_DIM + (255,), batch=static)
    L("INDOORS", x=560, y=228, font_size=11,
      color=INK_DIM + (255,), batch=static)
    for label, (lx, ly), (tx, ty) in LABELS_SIDE:       # leaders + labels
        S.Line(lx + 4, ly - 2, tx, ty, 1, (120, 124, 132), batch=static)
        for i, line_ in enumerate(label.split("\n")):
            L(line_, x=lx, y=ly - i * 13, font_size=8,
              color=INK + (255,), batch=static)
    # flip arc + label
    ra = DRUM["r"] + 42
    S.Arc(dcx, dcy, ra, start_angle=math.radians(25),
          angle=math.radians(160), thickness=2,
          color=FLOW_COLORS["expansion"], batch=static)
    L("180°", x=dcx + ra + 8, y=dcy + 110, font_size=10,
      color=FLOW_COLORS["expansion"] + (255,), batch=static)
    # dimension callouts
    S.Line(dcx - DRUM["r"], 300, dcx + DRUM["r"], 300, 1,
           INK_DIM, batch=static)
    L("Ø 540 mm rotating core", x=dcx, y=306, font_size=9,
      anchor_x="center", color=INK_DIM + (255,), batch=static)
    # scale bar
    S.Line(70, 92, 170, 92, 2, INK, batch=static)
    for i in range(3):
        S.Line(70 + i * 50, 87, 70 + i * 50, 97, 2, INK, batch=static)
        L(str(i * 50), x=70 + i * 50, y=103, font_size=8,
          anchor_x="center", color=INK_DIM + (255,), batch=static)
    L("mm", x=175, y=95, font_size=8, color=INK_DIM + (255,), batch=static)

    # ---- rotating drum interior (dynamic, rotates about drum center) ----
    def crect(d, col):
        r = S.Rectangle(d["x"] + d["w"] / 2, d["y"] + d["h"] / 2,
                        d["w"], d["h"], color=col, batch=batch)
        r.anchor_x, r.anchor_y = d["w"] / 2, d["h"] / 2
        return r

    drum_parts = []
    for d, col in ((COIL_A, (78, 82, 92)), (COIL_B, (78, 82, 92)),
                   (COMP, (88, 96, 90)), (EXP, (96, 80, 100)),
                   (TRAY, (70, 90, 100))):
        drum_parts.append(dict(
            sh=crect(d, col),
            c=(d["x"] + d["w"] / 2, d["y"] + d["h"] / 2)))
    for f in (FAN_A, FAN_B):
        drum_parts.append(dict(
            sh=S.Circle(f["cx"], f["cy"], f["r"], color=(60, 66, 74),
                        batch=batch),
            c=(f["cx"], f["cy"])))
    blades = {"A": [S.Line(FAN_A["cx"], FAN_A["cy"],
                           FAN_A["cx"] + FAN_A["r"], FAN_A["cy"], 3,
                           INK_DIM, batch=batch) for _ in range(4)],
              "B": [S.Line(FAN_B["cx"], FAN_B["cy"],
                           FAN_B["cx"] + FAN_B["r"], FAN_B["cy"], 3,
                           INK_DIM, batch=batch) for _ in range(4)]}

    # refrigerant loop lines (rotate with the drum in flip mode)
    seg_lines = [S.Line(0, 0, 1, 1, 4, INK, batch=batch)
                 for _ in loop_segments("cool")]
    path_pts, path_len = _path_table("cool")
    N_P = 56
    particles = [{"shape": S.Circle(0, 0, 4, color=(255, 255, 255),
                                    batch=batch),
                  "s": i * path_len / N_P} for i in range(N_P)]

    # telemetry column
    tele = [L("", x=1190, y=790 - i * 20, font_size=10,
              color=INK + (255,), batch=batch) for i in range(12)]
    L("TELEMETRY", x=1190, y=812, font_size=10,
      color=INK_DIM + (255,), batch=static)

    # ================= PAGE 2 — CYCLE (live log p–h diagram) =================
    rr = RefrigerantR32
    AX0, AX1, AY0, AY1 = 110, 950, 90, 770
    P_LO, P_HI = 0.12e6, 6.0e6

    def hx(h):
        return AX0 + (h - H_MIN) / (H_MAX - H_MIN) * (AX1 - AX0)

    def py_(p):
        return AY0 + (math.log10(max(p, 1e4)) - math.log10(P_LO)) / \
            (math.log10(P_HI) - math.log10(P_LO)) * (AY1 - AY0)

    L("CYCLE — log p / h diagram (live)", x=60, y=806, font_size=13,
      weight="bold", color=INK + (255,), batch=cyc_b)
    S.Line(AX0, AY0, AX0, AY1, 2, INK, batch=cyc_b)
    S.Line(AX0, AY0, AX1, AY0, 2, INK, batch=cyc_b)
    L("enthalpy h (kJ/kg)", x=(AX0 + AX1) / 2, y=58, font_size=10,
      anchor_x="center", color=INK_DIM + (255,), batch=cyc_b)
    L("log p (MPa)", x=20, y=770, font_size=10,
      color=INK_DIM + (255,), batch=cyc_b)

    # saturation dome from the actual refrigerant model
    TS_DOME = [235.0 + i * (RefrigerantR32.T_CRIT - 236.0) / 48.0
               for i in range(49)]
    dome_l = [(rr.h_liq(t), rr.psat(t)) for t in TS_DOME]
    dome_r = [(rr.h_g_sat(t), rr.psat(t)) for t in TS_DOME]
    H_MIN = min(h for h, _ in dome_l) - 30e3
    H_MAX = max(h for h, _ in dome_r) + 60e3
    for pts in (dome_l, dome_r):
        for i in range(len(pts) - 1):
            S.Line(hx(pts[i][0]), py_(pts[i][1]), hx(pts[i + 1][0]),
                   py_(pts[i + 1][1]), 2, (140, 145, 155), batch=cyc_b)
    for pm in (0.2, 0.5, 1.0, 2.0, 4.0):                 # pressure grid
        yy = py_(pm * 1e6)
        S.Line(AX0, yy, AX1, yy, 1, (82, 85, 93), batch=cyc_b)
        L(f"{pm:.1f}", x=58, y=yy - 5, font_size=9,
          color=INK_DIM + (255,), batch=cyc_b)
    for hk in range(int(H_MIN // 1e5) * 100, int(H_MAX // 1e5) * 100 + 1,
                    100):                                 # h grid (kJ/100)
        xx = hx(hk * 1e3)
        S.Line(xx, AY0, xx, AY1, 1, (82, 85, 93), batch=cyc_b)
        L(str(hk), x=xx, y=74, font_size=9, anchor_x="center",
          color=INK_DIM + (255,), batch=cyc_b)

    # live cycle overlay: 4 legs + 4 state points + state table
    cyc_ln = [S.Line(0, 0, 1, 1, 5, FLOW_COLORS[t], batch=cyc_b)
              for t in ("discharge", "condenser", "expansion", "evaporator")]
    cyc_pt = [S.Circle(0, 0, 9, color=INK, batch=cyc_b) for _ in range(4)]
    cyc_num = [L(str(i + 1), x=0, y=0, font_size=9,
                 anchor_x="center", color=BG_GREY + (255,), batch=cyc_b)
               for i in range(4)]
    state_names = ("1 suction → comp", "2 discharge", "3 liquid out",
                   "4 evap inlet")
    cyc_rows = [L("", x=1000, y=720 - i * 30, font_size=11,
                  color=INK + (255,), batch=cyc_b) for i in range(4)]
    cyc_meta = [L("", x=1000, y=560 - i * 26, font_size=11,
                  color=INK_DIM + (255,), batch=cyc_b) for i in range(6)]
    L("STATE POINTS", x=1000, y=760, font_size=11,
      color=INK_DIM + (255,), batch=cyc_b)
    L("ENERGY", x=1000, y=600, font_size=11,
      color=INK_DIM + (255,), batch=cyc_b)

    # ============== PAGES 3/4 — SPECS / VALIDATION (captured text) ==============
    for pg, title, fn in (
            (2, "ENGINEERING SPEC SHEET", lambda: spec_sheet(core)),
            (3, "VALIDATION — live harness output",
             lambda: validate(core, opt))):
        L(title, x=60, y=806, font_size=13, weight="bold",
          color=INK + (255,), batch=pages[pg])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn()
        lines = buf.getvalue().splitlines()
        for i, ln_ in enumerate(lines[:36]):
            L(ln_ if ln_.strip() else " ", x=60, y=770 - i * 21,
              font_size=11, color=INK + (255,), batch=pages[pg])
        if len(lines) > 36:
            L(f"… {len(lines) - 36} more lines (see --spec/--validate)",
              x=60, y=770 - 37 * 21, font_size=10,
              color=INK_DIM + (255,), batch=pages[pg])

    # ======= PAGE 5 — 3D ASSEMBLY (PyOpenGL, reference-code style) =======
    # Fixed-function pipeline: perspective camera + lighting + depth test,
    # parts drawn with glPushMatrix/glTranslatef — same idiom as the
    # reference observer windows. Units are mm; drum center is the origin.
    from OpenGL.GL import (glBegin, glEnd, glVertex3f, glNormal3f,
                           glColor4f, glClearColor, glClear, glViewport,
                           glMatrixMode, glLoadIdentity, glPushMatrix,
                           glPopMatrix, glTranslatef, glRotatef, glEnable,
                           glDisable, glLightfv, glDepthMask, glBlendFunc,
                           glColorMaterial,
                           glLineWidth, glGetDoublev, glGetIntegerv,
                           glOrtho, glUseProgram, glColor3f,
                           glDrawBuffer, glBindVertexArray,
                           glBindBuffer, glBindTexture, glActiveTexture,
                           glDisableClientState, GL_VERTEX_ARRAY,
                           GL_TEXTURE_COORD_ARRAY, GL_NORMAL_ARRAY,
                           GL_COLOR_ARRAY, GL_ARRAY_BUFFER,
                           GL_TEXTURE_2D, GL_TEXTURE0, GL_SCISSOR_TEST,
                           GL_QUADS, GL_LINES, GL_TRIANGLES, GL_PROJECTION,
                           GL_MODELVIEW, GL_DEPTH_TEST, GL_LIGHTING,
                           GL_LIGHT0, GL_LIGHT1, GL_POSITION, GL_DIFFUSE,
                           GL_AMBIENT, GL_COLOR_MATERIAL,
                           GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE,
                           GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                           GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                           GL_MODELVIEW_MATRIX, GL_PROJECTION_MATRIX,
                           GL_BACK, GL_VIEWPORT)
    from OpenGL.GLU import (gluPerspective, gluProject, gluNewQuadric,
                            gluCylinder, gluDisk)
    import OpenGL as _OGL
    _OGL.ERROR_CHECKING = False
    from ctypes import c_float as _cf
    quad3 = gluNewQuadric()
    RIM3, HW3 = 270.0, 110.0          # drum radius / half width (mm)

    C_FRAME = (96, 104, 116); C_WALL = (62, 65, 71); C_DRUM = (150, 160, 175)
    C_COILA = (92, 128, 160); C_COILB = (82, 150, 142); C_COMP = (112, 124, 96)
    C_EXP = (140, 104, 150); C_TRAY = (80, 128, 150); C_FAN = (185, 190, 200)
    C_BEAR = (125, 125, 135); C_PIPE = (205, 145, 90)

    def _c4(col, a=1.0):
        glColor4f(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0, a)

    def box3(x0, y0, z0, x1, y1, z1, col):
        _c4(col)
        glBegin(GL_QUADS)
        glNormal3f(0, 1, 0)
        glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0)
        glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1)
        glNormal3f(0, -1, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1)
        glVertex3f(x1, y0, z1); glVertex3f(x1, y0, z0)
        glNormal3f(1, 0, 0)
        glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0)
        glNormal3f(-1, 0, 0)
        glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0)
        glVertex3f(x0, y1, z1); glVertex3f(x0, y0, z1)
        glNormal3f(0, 0, 1)
        glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1)
        glVertex3f(x1, y1, z1); glVertex3f(x1, y0, z1)
        glNormal3f(0, 0, -1)
        glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0)
        glVertex3f(x1, y1, z0); glVertex3f(x0, y1, z0)
        glEnd()

    def tube3(a, b, r, col, slices=14):
        dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        ln_ = math.sqrt(dx * dx + dy * dy + dz * dz)
        if ln_ < 1e-6:
            return
        glPushMatrix()
        glTranslatef(a[0], a[1], a[2])
        ax_, ay_ = -dy, dx                      # cross((0,0,1), d)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, dz / ln_))))
        if abs(ax_) < 1e-9 and abs(ay_) < 1e-9:
            ax_, ang = (1.0, 180.0) if dz < 0 else (1.0, 0.0)
        glRotatef(ang, ax_, ay_, 0.0)
        _c4(col)
        gluCylinder(quad3, r, r, ln_, slices, 1)
        glPopMatrix()

    def fins3(x0, x1, y0, y1, z, n=9):
        glBegin(GL_LINES)
        for i in range(n):
            xx = x0 + (x1 - x0) * i / max(1, n - 1)
            glVertex3f(xx, y0, z); glVertex3f(xx, y1, z)
        glEnd()

    def drum_shell():
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthMask(0)
        glPushMatrix(); glTranslatef(-HW3, 0, 0); glRotatef(90, 0, 1, 0)
        _c4(C_DRUM, 0.14)
        gluCylinder(quad3, RIM3, RIM3, 2 * HW3, 44, 1)
        glPopMatrix()
        for sx in (-HW3, HW3):                   # rim rings + louvered caps
            glPushMatrix(); glTranslatef(sx, 0, 0)
            glRotatef(90.0 if sx > 0 else -90.0, 0, 1, 0)
            _c4(C_DRUM, 0.5)
            gluDisk(quad3, RIM3 - 4, RIM3, 44, 1)
            _c4(C_DRUM, 0.10)
            gluDisk(quad3, 0, RIM3 - 4, 44, 1)
            _c4(C_DRUM, 0.45)
            glBegin(GL_LINES)
            for k in range(12):                  # louver spokes
                a = k * math.pi / 6.0
                glVertex3f(20 * math.cos(a), 20 * math.sin(a), 0)
                glVertex3f((RIM3 - 10) * math.cos(a),
                           (RIM3 - 10) * math.sin(a), 0)
            glEnd()
            glPopMatrix()
        glDepthMask(1)
        glDisable(GL_BLEND)

    def axial_fan():
        glPushMatrix(); glTranslatef(0, 155, -110)
        _c4(C_FAN)
        gluDisk(quad3, 0, 16, 18, 1)             # hub
        glPushMatrix()
        glRotatef(math.degrees(state["fa_a"]), 0, 0, 1)
        for k in range(5):                       # pitched blades
            glPushMatrix()
            glRotatef(k * 72.0, 0, 0, 1)
            glRotatef(28, 1, 0, 0)
            box3(14, -7, -3, 58, 7, 3, C_FAN)
            glPopMatrix()
        glPopMatrix()
        _c4(C_DRUM)                              # shroud ring
        glBegin(GL_LINES)
        for k in range(33):
            a0 = k * math.pi / 16.5
            glVertex3f(64 * math.cos(a0), 64 * math.sin(a0), 0)
            a1 = (k + 1) * math.pi / 16.5
            glVertex3f(64 * math.cos(a1), 64 * math.sin(a1), 0)
        glEnd()
        glPopMatrix()

    def crossflow():
        glPushMatrix(); glTranslatef(0, 150, 120)
        glRotatef(math.degrees(state["fb_a"]), 1, 0, 0)
        glPushMatrix(); glTranslatef(-90, 0, 0); glRotatef(90, 0, 1, 0)
        _c4(C_FAN)
        gluCylinder(quad3, 40, 40, 180, 20, 1)
        glPopMatrix()
        for k in range(10):                      # rotor fins
            glPushMatrix()
            glRotatef(k * 36.0, 1, 0, 0)
            box3(-88, 32, -2, 88, 40, 2, C_FAN)
            glPopMatrix()
        glPopMatrix()

    parts3 = []   # (name, explode_offset_mm, inside_drum, draw_fn)

    def P(name, off, drum_, fn):
        parts3.append((name, off, drum_, fn))

    P("wall", (0, 0, 0), False, lambda: [
        box3(-355, 290, -50, 355, 305, 50, C_WALL),
        box3(-355, -305, -50, 355, -290, 50, C_WALL),
        box3(-355, -290, -50, -290, 290, 50, C_WALL),
        box3(290, -290, -50, 355, 290, 50, C_WALL)])
    P("frame top", (0, 240, 0), False,
      lambda: box3(-290, 272, -45, 290, 290, 45, C_FRAME))
    P("frame bottom", (0, -240, 0), False,
      lambda: box3(-290, -290, -45, 290, -272, 45, C_FRAME))
    P("frame left", (-360, 0, 0), False,
      lambda: box3(-290, -272, -45, -118, 272, 45, C_FRAME))
    P("frame right", (360, 0, 0), False,
      lambda: box3(118, -272, -45, 290, 272, 45, C_FRAME))
    P("bearing L", (-240, 0, 0), False,
      lambda: tube3((-145, 0, 0), (-112, 0, 0), 30, C_BEAR))
    P("bearing R", (240, 0, 0), False,
      lambda: tube3((112, 0, 0), (145, 0, 0), 30, C_BEAR))
    P("drum shell", (0, 0, 0), True, drum_shell)
    P("axle", (0, 0, 0), True,
      lambda: tube3((-120, 0, 0), (120, 0, 0), 16, (95, 100, 110)))
    P("coil A (outdoor)", (0, 0, -320), True, lambda: [
        box3(-100, -110, -208, 100, 110, -156, C_COILA),
        (_c4((150, 185, 215)), fins3(-96, 96, -105, 105, -155.5))])
    P("coil B (indoor)", (0, 0, 320), True, lambda: [
        box3(-100, -110, 156, 100, 110, 208, C_COILB),
        (_c4((150, 210, 200)), fins3(-96, 96, -105, 105, 208.5))])
    P("axial fan", (0, 240, -140), True, axial_fan)
    P("cross-flow fan", (0, 240, 140), True, crossflow)
    P("compressor", (0, -330, -30), True, lambda: [
        tube3((0, -225, 20), (0, -95, 20), 46, C_COMP),
        tube3((0, -225, 20), (0, -232, 20), 50, C_COMP)])   # feet ring
    P("expansion train", (-220, -200, 60), True,
      lambda: box3(-88, -215, 25, -28, -165, 85, C_EXP))
    P("condensate tray", (0, -220, 90), True,
      lambda: box3(-100, -250, -80, 100, -235, 80, C_TRAY))
    P("piping", (0, 0, 160), True, lambda: [
        tube3((0, -95, 20), (-58, -190, 55), 9, C_PIPE),     # comp → EEV
        tube3((-58, -190, 55), (-60, -60, -150), 9, C_PIPE),  # EEV → coil A
        tube3((60, -60, 150), (20, -140, 20), 9, C_PIPE),     # coil B → comp
        tube3((0, -170, -30), (0, -60, -156), 7, C_PIPE)])    # liquid line

    # floating part labels (projected with gluProject each frame)
    flabels = []  # (label, anchor, inside_drum, offset)
    for name, anchor, drum_, off in (
            ("coil A — outdoor", (0, 40, -185), True, (0, 0, -320)),
            ("coil B — indoor", (0, 40, 185), True, (0, 0, 320)),
            ("compressor", (0, -160, 20), True, (0, -330, -30)),
            ("axial fan", (0, 155, -110), True, (0, 240, -140)),
            ("cross-flow fan", (0, 150, 120), True, (0, 240, 140)),
            ("expansion train", (-58, -190, 55), True, (-220, -200, 60)),
            ("condensate tray", (0, -242, 0), True, (0, -220, 90)),
            ("drum rim (rotates 180°)", (0, 272, 0), False, (0, 0, 0)),
            ("adapter frame", (-290, 281, 0), False, (0, 240, 0)),
            ("bearing", (-128, 0, 0), False, (-240, 0, 0))):
        flabels.append((L(name, x=0, y=0, font_size=9,
                          color=INK + (255,), batch=leg_b),
                        anchor, drum_, off))

    # 3D HUD legend + status
    hud3 = [L("", x=40, y=790 - i * 20, font_size=11, weight="bold",
              color=INK + (255,), batch=leg_b) for i in range(3)]
    L("PARTS", x=40, y=215, font_size=10, color=INK_DIM + (255,),
      batch=leg_b)
    for i, (nm, col) in enumerate((
            ("drum shell", C_DRUM), ("coil A outdoor", C_COILA),
            ("coil B indoor", C_COILB), ("fans", C_FAN),
            ("compressor", C_COMP), ("expansion", C_EXP),
            ("tray", C_TRAY), ("frame", C_FRAME),
            ("bearings", C_BEAR), ("piping", C_PIPE))):
        S.Rectangle(40, 192 - i * 18, 10, 10, col, batch=leg_b)
        L(nm, x=56, y=192 - i * 18, font_size=8,
          color=INK_DIM + (255,), batch=leg_b)

    def render_3d():
        w3, h3 = win.width, win.height
        glViewport(0, 0, w3, h3)
        glDrawBuffer(GL_BACK)   # context can be left on GL_FRONT — the
        # back buffer is what flip() presents; drawing to front is invisible
        # pyglet 2.x leaves its shader program / VAO / font texture bound;
        # immediate-mode geometry is dropped or modulated while those are
        # active — fully reset to fixed-function defaults.
        glUseProgram(0)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glDisable(GL_TEXTURE_2D)
        glDisable(GL_SCISSOR_TEST)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        glClearColor(BG_GREY[0] / 255.0, BG_GREY[1] / 255.0,
                     BG_GREY[2] / 255.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(42.0, w3 / h3, 10.0, 9000.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        glTranslatef(0.0, -20.0, -state["zoom"])
        glRotatef(state["rx"], 1, 0, 0)
        glRotatef(state["ry"], 0, 1, 0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT0, GL_POSITION,
                  (_cf * 4)(3000.0, 4000.0, 5000.0, 0.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (_cf * 4)(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (_cf * 4)(0.30, 0.30, 0.32, 1.0))
        glLightfv(GL_LIGHT1, GL_POSITION,
                  (_cf * 4)(-4000.0, -1500.0, -3000.0, 0.0))
        glLightfv(GL_LIGHT1, GL_DIFFUSE, (_cf * 4)(0.35, 0.35, 0.40, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glLineWidth(1.0)
        flip = math.degrees(state["ang"])
        ex = state["expl"]
        for _nm, off, drum_, fn in parts3:
            glPushMatrix()
            if drum_:
                glRotatef(flip, 1, 0, 0)
            glTranslatef(off[0] * ex, off[1] * ex, off[2] * ex)
            fn()
            glPopMatrix()

        # project part labels to screen space
        mv = glGetDoublev(GL_MODELVIEW_MATRIX)
        pj = glGetDoublev(GL_PROJECTION_MATRIX)
        vp = glGetIntegerv(GL_VIEWPORT)
        ca, sa = math.cos(state["ang"]), math.sin(state["ang"])
        for lb, anchor, drum_, off in flabels:
            ax_, ay_, az_ = anchor
            if drum_:
                ay_, az_ = ay_ * ca - az_ * sa, ay_ * sa + az_ * ca
            try:
                sx, sy, sz = gluProject(ax_ + off[0] * ex,
                                        ay_ + off[1] * ex,
                                        az_ + off[2] * ex, mv, pj, vp)
            except Exception:
                sz = 2.0
            if 0.0 < sz < 1.0:
                mx_, my_ = to_model(sx, sy)   # project→model space so the
                lb.x, lb.y = mx_ + 8, my_ - 5  # letterboxed label batch aligns
                lb.color = lb.color[:3] + (230,)
            else:
                lb.color = lb.color[:3] + (0,)
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)

    def locate(s_arc):
        s_arc %= path_len
        for x1, y1, x2, y2, ln_, start, tag in path_pts:
            if start + ln_ >= s_arc:
                f = (s_arc - start) / max(ln_, 1e-9)
                return x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, tag
        return path_pts[-1][2], path_pts[-1][3], path_pts[-1][6]

    def current_segs():
        if core.actuator.method == "flip":
            out = []
            for x1, y1, x2, y2, t in loop_segments("cool"):
                ax, ay = rot(dcx, dcy, x1, y1, state["ang"])
                bx, by = rot(dcx, dcy, x2, y2, state["ang"])
                out.append((ax, ay, bx, by, t))
            return out
        return loop_segments(state["mode"])

    def rebuild_path():
        nonlocal path_len
        segs = current_segs()
        path_pts.clear()
        total = 0.0
        for x1, y1, x2, y2, tag in segs:
            ln_ = math.hypot(x2 - x1, y2 - y1)
            path_pts.append((x1, y1, x2, y2, ln_, total, tag))
            total += ln_
        path_len = total
        for ln_, (x1, y1, x2, y2, tag) in zip(seg_lines, segs):
            ln_.x, ln_.y, ln_.x2, ln_.y2 = x1, y1, x2, y2
            ln_.color = FLOW_COLORS[tag]

    def refresh():
        res = core.solve(state["mode"], K(26.0), K(35.0), state["pt"])
        state["res"] = res
        rebuild_path()
        # air arrows follow the season: winter exhaust is cold, supply hot
        for ln_ in exhaust_arrow:
            ln_.color = HOT if state["mode"] == "cool" else COLD
        for ln_ in supply_arrow:
            ln_.color = COLD if state["mode"] == "cool" else HOT
        vals = [
            f"mode        {res.mode} ({res.actuation})",
            f"N           {res.n_rps:.0f} rps",
            f"T_evap      {C(res.t_evap):6.1f} °C",
            f"T_cond      {C(res.t_cond):6.1f} °C",
            f"T_discharge {C(res.t_discharge):6.1f} °C",
            f"p_lo/hi     {res.p_low/1e6:.2f}/{res.p_high/1e6:.2f} MPa",
            f"m_dot       {res.m_dot_ref*1000:6.1f} g/s",
            f"Q delivered {res.q_capacity:6.0f} W",
            f"W total     {res.w_total:6.0f} W",
            f"COP         {res.cop:6.2f} ({100*res.carnot_fraction:.0f}% Carnot)",
            f"noise       {res.dba_indoor:.1f} dBA in",
            f"EVI         {'ON' if res.evi_active else 'off'}",
        ]
        for lbl, v in zip(tele, vals):
            lbl.text = v

        # CYCLE page: recompute the 4 state points and redraw the polygon
        sh, sc = ExpansionTrain.SUPERHEAT_K, ExpansionTrain.SUBCOOL_K
        t_eo = res.t_evap + sh
        t1 = t_eo + ExpansionTrain.IHX_EFF * max(
            0.0, (res.t_cond - sc) - t_eo)
        h1 = rr.h_vap(t1, res.p_low)
        h2 = rr.h_vap(res.t_discharge, res.p_high)
        h3 = rr.h_liq(res.t_cond - sc)
        h4 = h3 - (h1 - rr.h_vap(t_eo, res.p_low))     # post-IHX throttle
        cpts = [(h1, res.p_low, t1), (h2, res.p_high, res.t_discharge),
                (h3, res.p_high, res.t_cond - sc), (h4, res.p_low, res.t_evap)]
        for i, (h, p, t) in enumerate(cpts):
            hn, pn, _ = cpts[(i + 1) % 4]
            ln_ = cyc_ln[i]
            ln_.x, ln_.y, ln_.x2, ln_.y2 = hx(h), py_(p), hx(hn), py_(pn)
            cyc_pt[i].x, cyc_pt[i].y = hx(h), py_(p)
            cyc_num[i].x, cyc_num[i].y = hx(h) - 3, py_(p) - 4
            cyc_rows[i].text = (f"{state_names[i]:15s} "
                                f"T={C(t):6.1f}°C  p={p / 1e6:4.2f} MPa  "
                                f"h={h / 1000.0:6.1f} kJ/kg")
        mvals = [
            f"ṁ = {res.m_dot_ref * 1000:.1f} g/s",
            f"Q = {res.q_capacity:.0f} W   W = {res.w_total:.0f} W",
            f"COP = {res.cop:.2f} ({100 * res.carnot_fraction:.0f}% Carnot)",
            f"EVI {'ON' if res.evi_active else 'off'} · "
            f"{res.actuation} reversal",
            f"{res.dba_indoor:.1f} dBA indoor",
            "dome = R-32 sat. curve (model)",
        ]
        for lbl, v in zip(cyc_meta, mvals):
            lbl.text = v
        hud3[0].text = (f"3D ASSEMBLY — {res.mode} ({res.actuation}) · "
                        f"COP {res.cop:.2f} · {res.dba_indoor:.1f} dBA indoor")
        hud3[1].text = (f"N {res.n_rps:.0f} rps · Q {res.q_capacity:.0f} W · "
                        f"W {res.w_total:.0f} W")
        hud3[2].text = ("drag = orbit · wheel = zoom · x = explode · "
                        "c/h/f still apply live")
        hud3[2].color = INK_DIM + (255,)

    refresh()

    def update(dt):
        # pyglet 2.1.x workaround: a batch's vertex buffers only upload after
        # an attribute write — prime each vertex domain once, else untouched
        # static geometry never renders.
        if not state["primed"]:
            state["primed"] = True
            for s in primed:                 # force vertex upload per shape
                s.x = s.x
        state["ang_t"] = math.pi if (core.actuator.method == "flip" and
                                     state["mode"] == "heat") else 0.0
        if abs(state["ang"] - state["ang_t"]) > 1e-3:
            state["ang"] += (state["ang_t"] - state["ang"]) * \
                min(1.0, dt * 3.0)
            for part in drum_parts:
                nx, ny = rot(dcx, dcy, part["c"][0], part["c"][1],
                             state["ang"])
                part["sh"].x, part["sh"].y = nx, ny
                if hasattr(part["sh"], "rotation"):
                    part["sh"].rotation = math.degrees(state["ang"])
            rim_dot.x = dcx + DRUM["r"] * math.cos(state["ang"])
            rim_dot.y = dcy + DRUM["r"] * math.sin(state["ang"])
            rebuild_path()
        state["expl"] += (state["expl_t"] - state["expl"]) * \
            min(1.0, dt * 3.0)
        res = state["res"]
        for kkey, f, rpm in (("A", FAN_A, state["pt"].rpm_out),
                             ("B", FAN_B, state["pt"].rpm_in)):
            key = "fa_a" if kkey == "A" else "fb_a"
            state[key] += dt * rpm / 60.0 * 2 * math.pi * 0.2
            fcx, fcy = rot(dcx, dcy, f["cx"], f["cy"], state["ang"])
            for k, ln_ in enumerate(blades[kkey]):
                a = state[key] + k * math.pi / 2
                ln_.x, ln_.y = fcx, fcy
                ln_.x2 = fcx + f["r"] * 0.8 * math.cos(a)
                ln_.y2 = fcy + f["r"] * 0.8 * math.sin(a)
        if not state["paused"]:
            v = 40.0 + 3500.0 * res.m_dot_ref
            for p in particles:
                p["s"] += v * dt
                x, y, tag = locate(p["s"])
                p["shape"].x, p["shape"].y = x, y
                p["shape"].color = FLOW_COLORS[tag]
        state["accum"] += dt
        if state["accum"] > 0.6:
            state["accum"] = 0.0
            refresh()

    set_tab(0)

    @win.event
    def on_draw():
        win.clear()
        _apply_view()                        # letterbox 2D model space
        if state["tab"] == 4:
            render_3d()   # 3D does its own clear+depth; chrome overlays it
            chrome.draw()
            leg_b.draw()
        else:
            bg_b.draw()
            chrome.draw()
            pages[state["tab"]].draw()
            if state["tab"] == 0:
                batch.draw()

    @win.event
    def on_mouse_press(x, y, button, mods):
        mx, my = to_model(x, y)               # model space for hit tests
        if 852 <= my <= 886:                  # tab bar hit test
            for i, (_rr, _lb, x0, x1) in enumerate(tab_ui):
                if x0 <= mx <= x1:
                    set_tab(i)
                    return

    @win.event
    def on_mouse_drag(x, y, dx, dy, buttons, mods):
        _mx, my = to_model(x, y)
        if state["tab"] != 4 or my > 840 or not (buttons & _mouse.LEFT):
            return
        state["ry"] += dx * 0.5
        state["rx"] = max(-89.0, min(89.0, state["rx"] + dy * 0.5))

    @win.event
    def on_mouse_scroll(x, y, sx, sy):
        if state["tab"] == 4:
            state["zoom"] = max(650.0,
                                min(3200.0, state["zoom"] - sy * 70.0))

    @win.event
    def on_key_press(sym, mods):
        pt = state["pt"]
        if sym == _key.ESCAPE:
            if win.fullscreen:                # first ESC drops to windowed
                win.set_fullscreen(False)
            else:
                win.close()
            return
        if sym == _key.F11:                   # fullscreen toggle
            win.set_fullscreen(not win.fullscreen)
            return
        if sym == _key.TAB:
            set_tab((state["tab"] + 1) % len(TAB_NAMES))
            return
        for i, ks in enumerate((_key._1, _key._2, _key._3, _key._4,
                                _key._5)):
            if sym == ks:
                set_tab(i)
                return
        if sym == _key.X:
            state["expl_t"] = 0.0 if state["expl_t"] > 0.5 else 1.0
            return
        if sym == _key.SPACE:
            state["paused"] = not state["paused"]
        elif sym == _key.RIGHT:
            pt.n_rps = min(120.0, pt.n_rps + 5)
        elif sym == _key.LEFT:
            pt.n_rps = max(15.0, pt.n_rps - 5)
        elif sym == _key.UP:
            pt.rpm_in = min(900.0, pt.rpm_in + 50)
            pt.rpm_out = min(900.0, pt.rpm_out + 50)
        elif sym == _key.DOWN:
            pt.rpm_in = max(200.0, pt.rpm_in - 50)
            pt.rpm_out = max(250.0, pt.rpm_out - 50)
        elif sym == _key.C:
            state["mode"] = "cool"
        elif sym == _key.H:
            state["mode"] = "heat"
        elif sym == _key.F:
            core.actuator.method = "flip" if core.actuator.method == "valve" \
                else "valve"
        elif sym == _key.O:
            npt, _ = opt.best(state["mode"], K(26.0), K(35.0), 3200.0)
            state["pt"] = npt
        refresh()

    pyglet.clock.schedule_interval(update, 1.0 / 60.0)
    pyglet.app.run()


# =============================================================================
# §13  SELF-TEST / BENCH / CLI
# =============================================================================
# HONESTY LAYER: this model is correlation-grade. It reproduces the *shape*
# and approximate magnitude of real machine behavior (COP vs lift, noise vs
# rpm, part-load gains) but it is NOT a substitute for REFPROP + CFD + a
# bench prototype. Treat absolute numbers as ±10–15 % engineering estimates.

def self_test() -> None:
    r = RefrigerantR32
    # 1. vapor pressure sanity: R-32 ~0.79 MPa @ -3°C, ~1.7 MPa @ 27°C
    p1 = r.psat(K(-3.0)); p2 = r.psat(K(27.0))
    assert 0.65e6 < p1 < 0.95e6, f"psat(-3C)={p1:.3e}"
    assert 1.4e6 < p2 < 2.1e6, f"psat(27C)={p2:.3e}"
    # 2. tsat inversion
    assert abs(r.tsat(p2) - K(27.0)) < 0.5
    # 3. latent heat monotonic decrease toward Tc
    assert r.hfg(K(-20)) > r.hfg(K(20)) > r.hfg(K(57)) > 0
    # 4. entropy ordering: sat vapor > sat liquid
    assert r.s_g_sat(K(7)) > r.s_liq(K(7))
    # 5. cycle solves and COPs are sane
    core = HeatPumpCore()
    opt = QuietOptimizer(core)
    pt = OperatingPoint(90.0, 700.0, 600.0)
    cool = core.solve("cool", K(26.7), K(35.0), pt)
    assert cool.ok and 2.0 < cool.cop < 9.0, f"cool COP={cool.cop}"
    heat = core.solve("heat", K(21.0), K(8.3), pt)
    assert heat.ok and 2.0 < heat.cop < 8.0, f"heat COP={heat.cop}"
    cold = core.solve("heat", K(21.0), K(-15.0), pt)
    assert cold.evi_active and cold.cop > 1.5
    # 6. flip beats valve slightly (no valve Δp)
    flip = HeatPumpCore(actuation="flip")
    rf = flip.solve("cool", K(26.7), K(35.0), pt)
    assert rf.cop > cool.cop
    # 7. quiet optimizer finds a feasible, quieter point
    _, q = opt.best("cool", K(26.0), K(35.0), 3000.0)
    assert q.dba_indoor < 40.0, f"dba={q.dba_indoor}"
    # 8. dB summation
    assert abs(db_sum(30.0, 30.0) - 33.01) < 0.05
    # 9. condensate non-negative
    m_c, _ = condensate_rate(0.5, 30.0, 0.7, K(10.0))
    assert m_c >= 0.0
    # 10. frost accrues in the frosting window and derates UA
    f = FrostModel()
    for _ in range(120):
        f.update(60.0, -2.0, 0.9, False)
    assert f.level > 0.0 and f.ua_derate < 1.0
    # 11. IHX is near-neutral-to-helpful (within ±8% COP) and never crashes
    old_eff = ExpansionTrain.IHX_EFF
    ExpansionTrain.IHX_EFF = 0.0
    noihx = core.solve("cool", K(26.7), K(35.0), pt)
    ExpansionTrain.IHX_EFF = old_eff
    assert abs(cool.cop / noihx.cop - 1.0) < 0.08
    # 12. EVI genuinely adds capacity at −15 °C vs. same point without
    old_thr = core.evi_threshold_c
    core.evi_threshold_c = -99.0
    noevi = core.solve("heat", K(21.0), K(-15.0), pt)
    core.evi_threshold_c = old_thr
    assert cold.q_capacity > noevi.q_capacity
    # 13. latent split: 0 ≤ latent ≤ total delivered (cooling)
    assert 0.0 <= cool.q_latent <= cool.q_capacity + 1.0
    assert cool.q_sensible <= cool.q_capacity
    # 14. night cap is honored when feasible
    _, nres = opt.best("cool", K(26.0), K(30.0), 2200.0, dba_cap=25.0)
    assert nres.dba_indoor <= 25.0 + 1e-9
    # 15. arctic capability: at -50 °C the multi-port EVI + liquid-quench
    # path must still deliver net-positive heat with T_dis under the limit
    _, arc = opt.best("heat", K(21.0), K(-50.0), 1400.0)
    assert arc.ok and arc.cop > 1.0 and arc.q_capacity >= 1400.0, \
        f"arctic: cop={arc.cop} ok={arc.ok} q={arc.q_capacity}"
    print("self-test: all 15 checks passed")


def _bench() -> None:
    core = HeatPumpCore()
    opt = QuietOptimizer(core)
    t0 = time.time()
    pt, res = opt.best("cool", K(26.0), K(35.0), 3200.0)
    dt = time.time() - t0
    n = len(QuietOptimizer.N_GRID) * len(QuietOptimizer.RPM_IN_GRID) * \
        len(QuietOptimizer.RPM_OUT_GRID)
    print(f"bench: {n} grid solves in {dt:.2f}s "
          f"({n/dt:.0f} solves/s) → N={pt.n_rps}rps, "
          f"{res.dba_indoor:.1f} dBA, COP {res.cop:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AURA-1 window-replacement heat pump — physics replica")
    ap.add_argument("--test", action="store_true", help="run self-test suite")
    ap.add_argument("--bench", action="store_true", help="optimizer benchmark")
    ap.add_argument("--rate", action="store_true", help="rating points table")
    ap.add_argument("--season", action="store_true", help="SEER/HSPF bins")
    ap.add_argument("--quiet-compare", action="store_true")
    ap.add_argument("--day", choices=["cool", "heat"],
                    help="24-h closed-loop room sim")
    ap.add_argument("--night", action="store_true",
                    help="apply 25 dBA night cap (22:00–07:00) during --day")
    ap.add_argument("--flip", action="store_true",
                    help="use physical 180° flip actuator instead of valve")
    ap.add_argument("--spec", action="store_true", help="engineering spec sheet")
    ap.add_argument("--validate", action="store_true",
                    help="conservation audit + Carnot sweep + property check")
    ap.add_argument("--view", action="store_true",
                    help="interactive to-scale model window (pyglet)")
    ap.add_argument("--svg", action="store_true",
                    help="export to-scale drawing sheet to AURA-1_diagram.svg")
    ap.add_argument("--optimize", action="store_true",
                    help="design-space sweep: coil UA × fan size × Vd")
    ap.add_argument("--blueprint", action="store_true",
                    help="export frozen digital design package (JSON)")
    ap.add_argument("--all", action="store_true", help="run everything")
    args = ap.parse_args()
    # Windows consoles default to cp1252; symbols below need UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    actuation = "flip" if args.flip else "valve"
    core = HeatPumpCore(actuation=actuation)
    opt = QuietOptimizer(core)
    sim = ClosedLoopSimulator(core, opt)

    banner = (f"AURA-1 replica — R-32, inverter, ejector+EVI, "
              f"actuation={actuation} (quiet build)")
    print("=" * len(banner)); print(banner); print("=" * len(banner))

    if args.test or args.all:
        self_test()
    if args.validate or args.all:
        validate(core, opt)
    if args.bench:
        _bench()
    if args.spec or args.all:
        spec_sheet(core)
    if args.rate or args.all:
        rating_points(core, opt)
    if args.season or args.all:
        seasonal_estimate(core, opt)
    if args.quiet_compare or args.all:
        quiet_compare(core, opt)
    if args.optimize or args.all:
        design_sweep()
    if args.blueprint or args.all:
        print(f"\nblueprint → {export_blueprint(core, opt)}")
    if args.day or args.all:
        mode = args.day or "cool"
        if mode == "cool":
            prof = DayProfile(25.0, 38.0, rh=0.55)
            room = Room(t0=K(30.0))
            res = sim.run_day("cool", prof, K(24.0), room, label="hot day",
                              night_cap=25.0 if args.night else None)
        else:
            prof = DayProfile(-6.0, 3.0, rh=0.8)
            room = Room(t0=K(18.0))
            res = sim.run_day("heat", prof, K(21.0), room, label="cold day",
                              night_cap=25.0 if args.night else None)
        print(f"\n--- {res['label']} summary ---")
        print(f"  HVAC energy : {res['energy']['hvac_kwh']:.2f} kWh "
              f"(resistance-heat baseline {res['energy']['resist_kwh']:.2f} kWh)")
        print(f"  Defrost     : {res['energy']['defrost_kwh']:.2f} kWh")
        print(f"  Avg COP     : {res['cop_avg']:.2f}   "
              f"Avg capacity  : {res['cap_avg_w']:.0f} W")
        print(f"  Indoor noise: avg {res['dba_avg']:.1f} dBA, "
              f"max {res['dba_max']:.1f} dBA")
    if args.svg or args.all:
        res = core.solve("cool", K(26.0), K(35.0), OperatingPoint(60, 500, 500))
        print(f"\ndiagram → {export_svg('AURA-1_diagram.svg', res)}")
    # bare `python AC.py` (and --view) launches the tabbed model window
    if args.view or not any(vars(args).values()):
        run_viewer(core, opt)


if __name__ == "__main__":
    main()
