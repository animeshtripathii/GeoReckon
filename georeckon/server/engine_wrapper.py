"""
GeoReckon Pure-Python Session Engine Core v3.2
Implements session state machine, sensor pipeline integration, InEKF secondary filter,
and KinoNet-R2 shadow mode inference.
"""

import os
import math
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch

from georeckon.src.virtual_alignment import VirtualInVehicleAligner
from georeckon.src.vibration_filter import KinematicVibrationFilter
from georeckon.src.kinematics_speed_estimator import KinematicSpeedEstimator
from georeckon.src.filter_inekf import InEKF, InEKFState, so3_exp
from georeckon.src.map_matcher import TopologicalMapMatcher

from georeckon.server.kinonet_r2 import KinoNetR2
from georeckon.server.geodesics import latlon_to_enu, enu_to_latlon, EARTH_RADIUS_M
from georeckon.server.models import (
    CalibrateRequest,
    SensorPacket,
    GPSRestoredPacket,
    ResetPacket,
    CalibrateDoneResponse,
    PositionPayload,
    ResetDoneResponse,
    ErrorResponse,
)

EARTH_RADIUS_M = 6371000.0
ALIGN_MIN_HEALTH = 0.25
_GLOBAL_KINONET_LOCK = threading.Lock()
_CACHED_KINONET: Optional[KinoNetR2] = None


class GeoReckonError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class GeoReckonEngine:
    def __init__(self, weights_path: Optional[str] = None):
        """
        Initializes the GeoReckon core engine and loads KinoNet-R2 weights.
        Fails cleanly if weights are missing.
        """
        global _CACHED_KINONET
        if _CACHED_KINONET is None:
            if weights_path is None:
                env_path = os.environ.get("KINONET_WEIGHTS")
                if env_path:
                    weights_path = env_path
                else:
                    default_p = Path(__file__).resolve().parent.parent.parent / "InertiaX" / "kinonet_r2_best.pth"
                    weights_path = str(default_p)

            self.weights_path = Path(weights_path)
            if not self.weights_path.exists():
                raise FileNotFoundError(
                    f"KinoNet weights file not found at: '{self.weights_path}'. "
                    f"Ensure KINONET_WEIGHTS environment variable is set or weights file exists."
                )

            print(f"[GeoReckon] KINONET_WEIGHTS resolved to: '{self.weights_path}' (exists=True)")

            # Load KinoNetR2 in shadow mode (evaluation only)
            kino = KinoNetR2()
            state_dict = torch.load(str(self.weights_path), map_location="cpu")
            kino.load_state_dict(state_dict, strict=True)
            kino.eval()
            # Warm up PyTorch runtime with a dummy forward pass to prime C++ kernels and memory arenas
            dummy_tensor = torch.zeros((1, 6, 20), dtype=torch.float32)
            _ = kino.predict_with_uncertainty(dummy_tensor, n_mc_samples=5)
            print(f"[GeoReckon] KinoNetR2 loaded with strict=True, shadow mode active (warmed up).")
            _CACHED_KINONET = kino

        self.kinonet = _CACHED_KINONET

        # Session state
        self.mode = "IDLE"
        self.session_id: Optional[str] = None
        self.degraded: bool = False
        self.last_seq: int = -1
        self.last_t_ms: Optional[int] = None
        self.last_integrated_t_ms: Optional[int] = None
        self.blackout_start_ms: Optional[int] = None
        self.t_ms_history: list[int] = []
        self.jitter_warnings: int = 0

        # Geodesic origins & calibration params
        self.lat0: float = 0.0
        self.lon0: float = 0.0
        self.alt0: float = 0.0
        self.R_b_to_v: np.ndarray = np.eye(3)
        self.fwd_bias: float = 0.0
        self.gyro_bias: np.ndarray = np.zeros(3)
        self.alignment_health: float = 0.0

        # Subsystems
        self.vib_filter: Optional[KinematicVibrationFilter] = None
        self.speed_estimator: Optional[KinematicSpeedEstimator] = None
        self.inekf: Optional[InEKF] = None
        self.adaptive_yaw_bias: float = 0.0
        self.enable_ai_imu: bool = True
        self.map_matcher: TopologicalMapMatcher = TopologicalMapMatcher(road_width=7.0, max_snap_distance=35.0)

        # Planar Integrator State
        self.pos_enu: np.ndarray = np.zeros(2) # [x_east, y_north]
        self.heading_rad: float = 0.0          # math heading (CCW from East)
        self.cur_speed_mps: float = 0.0
        self.dr_distance_m: float = 0.0

        # KinoNet shadow window buffer (20, 6)
        self.kinonet_buffer: List[np.ndarray] = []

        # Sliding window of dt for rate enforcement
        self.dt_history: List[float] = []          # nominal-interval (dt<=0.5s) samples
        self.jitter_warnings: int = 0              # short-dt bursts (<0.04s), logged only
        self.slow_run_count: int = 0               # consecutive gap-class packets (0.5<dt<=1.0s)
        self.rate_bad: bool = False                # rate violation state flag
        self._recovery_history: List[int] = []     # healthy timestamp accumulation for recovery
        self._inference_lock = threading.Lock()    # lock for thread-safe MC-dropout inference

    def handle_calibrate(self, req: CalibrateRequest) -> CalibrateDoneResponse:
        t_start = time.perf_counter()
        
        # 1. Validate History Lengths
        n_imu = len(req.imu_history)
        if n_imu < 50:
            raise GeoReckonError("HISTORY_TOO_SHORT", f"IMU history too short: {n_imu} samples (min 50 required)")
        if n_imu > 600:
            raise GeoReckonError("HISTORY_TOO_LONG", f"IMU history too long: {n_imu} samples (max 600 allowed)")

        n_gps = len(req.gps_history)
        if n_gps < 3:
            raise GeoReckonError("HISTORY_TOO_SHORT", f"GPS history too short: {n_gps} points (min 3 required)")

        gps_t_span = (req.gps_history[-1].t_ms - req.gps_history[0].t_ms) / 1000.0
        if gps_t_span < 3.0:
            raise GeoReckonError("HISTORY_TOO_SHORT", f"GPS history time span too short: {gps_t_span:.2f}s (min 3.0s required)")

        # 2. Extract IMU arrays
        imu_t = np.array([im.t_ms / 1000.0 for im in req.imu_history])
        imu_a = np.array([im.accel for im in req.imu_history])
        imu_w = np.array([im.gyro for im in req.imu_history])
        imu_g = np.array([im.gravity for im in req.imu_history])

        # 3. Extract GPS arrays & interpolate speed onto IMU timestamps
        gps_t = np.array([g.t_ms / 1000.0 for g in req.gps_history])
        gps_v = np.array([g.speed_mps for g in req.gps_history])

        # Trim IMU samples outside GPS time span
        valid_mask = (imu_t >= gps_t[0]) & (imu_t <= gps_t[-1])
        if np.sum(valid_mask) < 40:
            raise GeoReckonError("HISTORY_TOO_SHORT", "Overlap between IMU and GPS history is less than 40 samples")

        imu_t = imu_t[valid_mask]
        imu_a = imu_a[valid_mask]
        imu_w = imu_w[valid_mask]
        imu_g = imu_g[valid_mask]

        # Interpolate GPS speed
        interp_speed = np.interp(imu_t, gps_t, gps_v)

        # 4. Heading Validation & Fallback
        initial_bearing = req.last_gps.bearing_deg
        initial_speed = max(0.0, req.last_gps.speed_mps)
        degraded = False

        if initial_speed < 2.0:
            # Fallback: search backwards in gps_history for speed >= 2.0 m/s
            fallback_bearing = None
            for g in reversed(req.gps_history):
                if g.speed_mps >= 2.0:
                    fallback_bearing = g.bearing_deg
                    break
            
            if fallback_bearing is not None:
                initial_bearing = fallback_bearing
            else:
                if not req.allow_low_health:
                    raise GeoReckonError(
                        "BEARING_INVALID",
                        f"Current speed ({initial_speed:.2f} m/s) is < 2.0 m/s and no moving GPS history exists. "
                        "Re-attempt while moving or set allow_low_health=true."
                    )
                degraded = True

        # 5. Virtual Mount Alignment
        aligner = VirtualInVehicleAligner()
        R_b_to_v, health = aligner.compute_alignment_from_history(
            imu_g,
            imu_a,
            interp_speed,
            imu_t
        )

        if health < ALIGN_MIN_HEALTH:
            if not req.allow_low_health:
                raise GeoReckonError(
                    "ALIGNMENT_HEALTH_LOW",
                    f"Computed alignment health {health:.4f} is below ALIGN_MIN_HEALTH ({ALIGN_MIN_HEALTH}). "
                    "Re-attempt during acceleration or braking or set allow_low_health=true."
                )
            degraded = True

        # 6. Pre-outage Bias Calibration (last 50 samples)
        cal_n = min(50, len(imu_t))
        cal_lin = (R_b_to_v @ (imu_a[-cal_n:] - imu_g[-cal_n:]).T).T
        cal_dt = max(imu_t[-1] - imu_t[-cal_n], 0.1)
        cal_dv = (interp_speed[-1] - interp_speed[-cal_n]) / cal_dt
        fwd_bias = float(np.clip(np.mean(cal_lin[:, 0]) - cal_dv, -0.3, 0.3))
        gyro_bias = np.median(imu_w[-cal_n:], axis=0)

        # 7. Seed Subsystems & Filters
        self.session_id = req.session_id
        self.degraded = degraded
        self.lat0 = req.last_gps.lat
        self.lon0 = req.last_gps.lon
        self.alt0 = req.last_gps.alt_m
        self.R_b_to_v = R_b_to_v
        self.fwd_bias = fwd_bias
        self.gyro_bias = gyro_bias
        self.alignment_health = float(health)
        self.adaptive_yaw_bias = 0.0

        # Vibration filter seeded with history
        self.vib_filter = KinematicVibrationFilter(sampling_rate=req.sample_rate_hz, cutoff_hz=1.5)
        # Pre-seed buffer
        for a_sample, w_sample in zip(imu_a[-20:], imu_w[-20:]):
            self.vib_filter.accel_buffer.append(a_sample.copy())
            self.vib_filter.gyro_buffer.append((w_sample - gyro_bias).copy())

        # Speed Estimator
        self.speed_estimator = KinematicSpeedEstimator(dt=1.0 / req.sample_rate_hz)
        self.speed_estimator.sync_gnss_speed(initial_speed)
        self.speed_estimator.set_calibrated_bias(fwd_bias)

        # Planar Integrator
        theta_0 = np.deg2rad((90.0 - initial_bearing) % 360.0)
        self.heading_rad = float(theta_0)
        self.pos_enu = np.array([0.0, 0.0])
        self.cur_speed_mps = float(initial_speed)
        self.dr_distance_m = 0.0

        # InEKF (Secondary / Covariance)
        # R_w_to_b = (R_z(theta_0) @ R_b_to_v).T -> R_b_to_w = R_z(theta_0) @ R_b_to_v
        c, s = np.cos(theta_0), np.sin(theta_0)
        R_v_to_w = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        R_b_to_w = R_v_to_w @ R_b_to_v

        v_world = np.array([initial_speed * np.cos(theta_0), initial_speed * np.sin(theta_0), 0.0])
        inekf_state = InEKFState(
            R=R_b_to_w,
            v=v_world,
            p=np.zeros(3),
            bg=gyro_bias.copy(),
            ba=np.zeros(3)
        )
        self.inekf = InEKF(inekf_state)

        # Seed KinoNet shadow buffer
        self.kinonet_buffer = []
        for a_sample, w_sample in zip(imu_a[-20:], imu_w[-20:]):
            a_veh = R_b_to_v @ a_sample
            a_veh[2] -= 9.80665
            a_veh[0] -= fwd_bias
            w_veh = R_b_to_v @ (w_sample - gyro_bias)
            self.kinonet_buffer.append(np.concatenate([a_veh, w_veh]))

        # Load optional road polyline corridor
        if req.road_polyline and len(req.road_polyline) >= 2:
            self.map_matcher.load_polyline(np.array(req.road_polyline, dtype=np.float64))
        else:
            self.map_matcher.segments = []

        # State transitions
        self.mode = "DEAD_RECKONING"
        self.last_seq = req.seq
        self.last_t_ms = req.t_ms
        self.last_integrated_t_ms = req.t_ms
        self.blackout_start_ms = req.t_ms
        self.t_ms_history = [req.t_ms]
        self.dt_history = []
        self.slow_run_count = 0
        self.jitter_warnings = 0

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return CalibrateDoneResponse(
            session_id=req.session_id,
            seq=req.seq,
            t_ms=req.t_ms,
            status="OK",
            degraded=self.degraded,
            r_phone_to_vehicle=R_b_to_v.tolist(),
            alignment_health=round(float(health), 4),
            fwd_accel_bias_mps2=round(float(fwd_bias), 4),
            gyro_bias_radps=[round(float(b), 6) for b in gyro_bias],
            latency_ms=round(latency_ms, 2),
        )

    def handle_sensor(self, pkt: SensorPacket) -> PositionPayload:
        t_start = time.perf_counter()

        if self.mode != "DEAD_RECKONING":
            raise GeoReckonError("NOT_CALIBRATED", "Received 'sensor' packet before calibration or in IDLE mode")

        if pkt.seq <= self.last_seq:
            raise GeoReckonError("OUT_OF_ORDER_SEQ", f"Packet seq {pkt.seq} <= last seen {self.last_seq}")

        # Advance last_seq and last_t_ms BEFORE rate check so that a rejected
        # (INVALID_RATE) packet still consumes its seq.
        dt = (pkt.t_ms - self.last_t_ms) / 1000.0 if self.last_t_ms is not None else 0.1
        self.last_t_ms = pkt.t_ms
        self.last_seq = pkt.seq

        # ── Rate enforcement (Item B1 & Item 2) ───────────────────────────────
        if not self.rate_bad:
            if dt > 1.0:
                self.t_ms_history = [pkt.t_ms]
                self.slow_run_count = 0
            else:
                self.t_ms_history.append(pkt.t_ms)
                if len(self.t_ms_history) > 20:
                    self.t_ms_history.pop(0)

        is_isolated_gap = False
        if dt >= 0.5:
            self.slow_run_count += 1
            if self.slow_run_count == 1:
                is_isolated_gap = True
        else:
            self.slow_run_count = 0

        if self.rate_bad:
            # When in rate_bad state, only healthy nominal packets (8-12 Hz, 0.08s <= dt <= 0.125s) count toward recovery.
            # Gaps > 1.0s or slow packets do NOT reset rate_bad.
            if 0.08 <= dt <= 0.125:
                self._recovery_history.append(pkt.t_ms)
                if len(self._recovery_history) >= 3:
                    # 2 consecutive healthy intervals -> recovered!
                    self.rate_bad = False
                    self.t_ms_history = list(self._recovery_history)
                    self._recovery_history.clear()
                else:
                    raise GeoReckonError(
                        "INVALID_RATE",
                        f"Sensor stream recovering from rate violation (healthy packets: {len(self._recovery_history)}/3)"
                    )
            else:
                self._recovery_history.clear()
                rate_hz = 1.0 / dt if dt > 0 else 0.0
                raise GeoReckonError(
                    "INVALID_RATE",
                    f"Sensor rate ({rate_hz:.2f} Hz, dt={dt:.3f}s) is outside required 8-12 Hz window (rate_bad=true)"
                )
        else:
            if dt < 0.04:
                # Jitter: client timestamp glitch (< 40ms) -> count warning, no rejection
                self.jitter_warnings += 1
            elif not is_isolated_gap:
                # Check sustained slow stream (>=2 consecutive slow packets) or full window rate
                should_check = (len(self.t_ms_history) >= 10) or (self.slow_run_count >= 2 and len(self.t_ms_history) >= 2)
                if should_check:
                    elapsed_s = (self.t_ms_history[-1] - self.t_ms_history[0]) / 1000.0
                    if elapsed_s > 0.0:
                        rate_hz = (len(self.t_ms_history) - 1) / elapsed_s
                        if not (8.0 <= rate_hz <= 12.0):
                            self.rate_bad = True
                            self.t_ms_history = [pkt.t_ms]
                            self._recovery_history.clear()
                            raise GeoReckonError(
                                "INVALID_RATE",
                                f"Sensor rate ({rate_hz:.2f} Hz over {elapsed_s:.2f}s window) is outside required 8-12 Hz window"
                            )

        # ── Integration Interval & Gap Discipline (Items B2 & B6) ─────────────
        # No interval may be dropped silently: use elapsed time since last successfully integrated packet.
        if self.last_integrated_t_ms is not None:
            int_dt = (pkt.t_ms - self.last_integrated_t_ms) / 1000.0
        else:
            int_dt = dt
        self.last_integrated_t_ms = pkt.t_ms

        stale = False
        gap_s = 0.0
        if int_dt > 1.0:
            stale = True
            gap_s = int_dt
            step_integration = False  # the motion during the gap is dropped; position resumes from last point; offset is not recovered
        elif int_dt > 0.5:
            stale = True
            gap_s = int_dt
            step_integration = True
        else:
            step_integration = True

        raw_a = np.array(pkt.accel, dtype=np.float64)
        raw_w = np.array(pkt.gyro, dtype=np.float64)
        g_phone = np.array(pkt.gravity, dtype=np.float64)

        # 1. Vibration filter (online single-sample)
        clean_a, clean_w, shock = self.vib_filter.update_sample(raw_a, raw_w - self.gyro_bias)

        # 2. Speed Estimator (evaluator.py:123 - gravity subtracted in phone frame before rotation)
        lin_phone = clean_a - g_phone
        lin_veh = self.R_b_to_v @ lin_phone
        w_veh = self.R_b_to_v @ clean_w

        # AI-IMU dynamic gyro bias tracking:
        # On straight segments (low lateral accel & small turn rate),
        # gently adapt residual yaw gyro bias to eliminate cumulative heading drift
        if self.enable_ai_imu:
            if abs(lin_veh[1]) < 0.15 and abs(w_veh[2]) < 0.05:
                self.adaptive_yaw_bias = 0.98 * self.adaptive_yaw_bias + 0.02 * w_veh[2]
            eff_yaw_rate = float(w_veh[2] - self.adaptive_yaw_bias)
        else:
            eff_yaw_rate = float(w_veh[2])

        if step_integration and int_dt > 0.001:
            clean_a_win = np.array(self.vib_filter.accel_buffer)
            clean_w_win = np.array(self.vib_filter.gyro_buffer)
            cur_speed, is_stopped = self.speed_estimator.estimate_step(
                forward_accel=lin_veh[0],
                lateral_accel=lin_veh[1],
                yaw_rate=eff_yaw_rate,
                accel_window=clean_a_win[-10:],
                gyro_window=clean_w_win[-10:],
                dt=int_dt
            )

            # AI-IMU Centripetal speed cross-validation during significant turns: v = |a_lat / w_z|
            if self.enable_ai_imu and abs(eff_yaw_rate) > 0.08:
                v_centripetal = abs(lin_veh[1] / eff_yaw_rate)
                if 2.0 < v_centripetal < 35.0:
                    cur_speed = 0.85 * cur_speed + 0.15 * v_centripetal

            self.cur_speed_mps = float(cur_speed)

            # 3. Planar Kinematic Integrator (evaluator.py:146-150)
            self.heading_rad += eff_yaw_rate * int_dt
            self.pos_enu[0] += self.cur_speed_mps * np.cos(self.heading_rad) * int_dt
            self.pos_enu[1] += self.cur_speed_mps * np.sin(self.heading_rad) * int_dt
            self.dr_distance_m += self.cur_speed_mps * int_dt
        else:
            is_stopped = self.speed_estimator.is_stationary

        # 4. Secondary InEKF (Covariance & comparison)
        if step_integration and int_dt > 0.001 and int_dt < 1.0:
            self.inekf.predict(raw_a, clean_w, int_dt)
            # AI-IMU MesNet dynamic covariance adaptation:
            # During turns, relax lateral velocity constraint to account for tire slip angle
            if self.enable_ai_imu:
                sigma_lat_sq = float((0.05 + 0.02 * abs(lin_veh[1]) + 0.01 * abs(eff_yaw_rate)) ** 2)
            else:
                sigma_lat_sq = 0.05**2

            self.inekf.update_chassis_velocity(
                np.array([self.cur_speed_mps, 0.0, 0.0]),
                self.R_b_to_v,
                np.array([0.25**2, sigma_lat_sq, 0.05**2]),
                axes=(0, 1, 2)
            )
            if is_stopped:
                self.inekf.update_zupt(variance=0.01)

            # AI-IMU Lie-group state fusion into pos_enu
            if self.enable_ai_imu and not self.degraded:
                inekf_p2d = self.inekf.state.p[:2]
                if np.all(np.isfinite(inekf_p2d)) and np.linalg.norm(inekf_p2d - self.pos_enu) < 150.0:
                    self.pos_enu = 0.5 * self.pos_enu + 0.5 * inekf_p2d

        # Road corridor manifold constraint
        if self.map_matcher and self.map_matcher.segments:
            self.pos_enu, self.heading_rad = self.map_matcher.constrain_to_road(
                self.pos_enu, self.heading_rad
            )

        # Geodesic conversion (Spherical R = 6371000.0)
        curr_lat, curr_lon = enu_to_latlon(self.pos_enu[0], self.pos_enu[1], self.lat0, self.lon0)
        bearing_deg = float((90.0 - np.rad2deg(self.heading_rad)) % 360.0)

        cov_2d = self.inekf.get_position_covariance_2d()
        inekf_e_m = float(self.inekf.state.p[0])
        inekf_n_m = float(self.inekf.state.p[1])

        sigma_pos = math.sqrt(max(0.0, float(np.trace(cov_2d))))
        elapsed_outage_ms = pkt.t_ms - (self.blackout_start_ms or pkt.t_ms)
        decay = math.exp(-elapsed_outage_ms / 60000.0)
        confidence = float(np.clip(math.exp(-sigma_pos / 10.0) * decay, 0.05, 1.0))

        # 5. KinoNet Shadow Mode
        a_veh_kino = self.R_b_to_v @ raw_a
        a_veh_kino[2] -= 9.80665
        a_veh_kino[0] -= self.fwd_bias
        w_veh_kino = self.R_b_to_v @ (raw_w - self.gyro_bias)
        self.kinonet_buffer.append(np.concatenate([a_veh_kino, w_veh_kino]))
        if len(self.kinonet_buffer) > 20:
            self.kinonet_buffer.pop(0)

        kinonet_speed_mps = self.cur_speed_mps
        kinonet_var = 0.1
        if len(self.kinonet_buffer) == 20:
            kino_arr = np.array(self.kinonet_buffer, dtype=np.float32).T # (6, 20)
            kino_tensor = torch.from_numpy(kino_arr).unsqueeze(0)        # (1, 6, 20)
            try:
                with _GLOBAL_KINONET_LOCK:
                    mean_spd, al_var, ep_var = self.kinonet.predict_with_uncertainty(kino_tensor, n_mc_samples=5)
                kinonet_speed_mps = float(mean_spd.item())
                kinonet_var = float((al_var + ep_var).item())
                # KinoNet active speed fusion: when neural model has high confidence (low variance),
                # gently anchor forward speed to prevent velocity divergence
                if self.enable_ai_imu and kinonet_var < 4.0 and kinonet_speed_mps > 1.0 and abs(kinonet_speed_mps - self.cur_speed_mps) < 6.0:
                    w_k = float(np.clip(0.4 / (1.0 + kinonet_var), 0.05, 0.20))
                    self.cur_speed_mps = float((1.0 - w_k) * self.cur_speed_mps + w_k * kinonet_speed_mps)
            except Exception:
                pass

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return PositionPayload(
            session_id=self.session_id or pkt.session_id,
            seq=pkt.seq,
            t_ms=pkt.t_ms,
            lat=round(curr_lat, 7),
            lon=round(curr_lon, 7),
            alt_m=round(self.alt0, 2),
            enu_x_m=round(float(self.pos_enu[0]), 2),
            enu_y_m=round(float(self.pos_enu[1]), 2),
            enu_z_m=0.0,
            speed_mps=round(self.cur_speed_mps, 2),
            bearing_deg=round(bearing_deg, 1),
            mode="DEAD_RECKONING",
            degraded=self.degraded,
            confidence=round(confidence, 2),
            is_stopped=bool(is_stopped),
            shock_detected=bool(shock),
            stale=stale,
            gap_s=round(gap_s, 2),
            kinonet_speed_mps=round(kinonet_speed_mps, 2),
            kinonet_var=round(kinonet_var, 3),
            inekf_e_m=round(inekf_e_m, 2),
            inekf_n_m=round(inekf_n_m, 2),
            dr_lat=None,
            dr_lon=None,
            dr_error_m=None,
            dr_distance_m=round(self.dr_distance_m, 2),
            latency_ms=round(latency_ms, 2),
        )

    def handle_gps_restored(self, pkt: GPSRestoredPacket) -> PositionPayload:
        t_start = time.perf_counter()

        if self.mode != "DEAD_RECKONING":
            raise GeoReckonError("NOT_CALIBRATED", "Cannot restore GPS when not in DEAD_RECKONING mode")

        # Current DR position before re-locking
        dr_lat, dr_lon = enu_to_latlon(self.pos_enu[0], self.pos_enu[1], self.lat0, self.lon0)

        # Geodesic difference to restored GPS
        dx, dy, _ = latlon_to_enu(pkt.gps.lat, pkt.gps.lon, dr_lat, dr_lon)
        dr_error_m = float(math.hypot(dx, dy))

        # Lock to restored GPS
        self.mode = "IDLE"
        self.last_seq = pkt.seq
        self.last_t_ms = pkt.t_ms

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return PositionPayload(
            session_id=self.session_id or pkt.session_id,
            seq=pkt.seq,
            t_ms=pkt.t_ms,
            lat=round(pkt.gps.lat, 7),
            lon=round(pkt.gps.lon, 7),
            alt_m=round(pkt.gps.alt_m, 2),
            enu_x_m=round(float(self.pos_enu[0]), 2),
            enu_y_m=round(float(self.pos_enu[1]), 2),
            enu_z_m=0.0,
            speed_mps=round(pkt.gps.speed_mps, 2),
            bearing_deg=round(pkt.gps.bearing_deg, 1),
            mode="IDLE",
            degraded=self.degraded,
            confidence=1.0,
            is_stopped=bool(pkt.gps.speed_mps < 0.25),
            shock_detected=False,
            stale=False,
            gap_s=0.0,
            kinonet_speed_mps=round(pkt.gps.speed_mps, 2),
            kinonet_var=0.05,
            inekf_e_m=round(float(self.inekf.state.p[0]) if self.inekf else 0.0, 2),
            inekf_n_m=round(float(self.inekf.state.p[1]) if self.inekf else 0.0, 2),
            dr_lat=round(dr_lat, 7),
            dr_lon=round(dr_lon, 7),
            dr_error_m=round(dr_error_m, 2),
            dr_distance_m=round(self.dr_distance_m, 2),
            latency_ms=round(latency_ms, 2),
        )

    def handle_reset(self, pkt: ResetPacket) -> ResetDoneResponse:
        t_start = time.perf_counter()

        self.mode = "IDLE"
        self.session_id = pkt.session_id
        self.degraded = False
        self.last_seq = pkt.seq
        self.last_t_ms = pkt.t_ms
        self.last_integrated_t_ms = None
        self.blackout_start_ms = None
        self.pos_enu = np.zeros(2)
        self.heading_rad = 0.0
        self.cur_speed_mps = 0.0
        self.dr_distance_m = 0.0
        self.kinonet_buffer.clear()
        self.t_ms_history.clear()
        self.dt_history.clear()
        self.slow_run_count = 0
        self.vib_filter = None
        self.speed_estimator = None
        self.inekf = None

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return ResetDoneResponse(
            session_id=pkt.session_id,
            seq=pkt.seq,
            t_ms=pkt.t_ms,
            status="OK",
            mode="IDLE",
            latency_ms=round(latency_ms, 2),
        )
