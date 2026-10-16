"""
GeoReckon WebSocket Protocol Models & Strict Validators v3.2
Implements Pydantic/dataclass serialization and schema enforcement.
"""

from typing import List, Optional, Dict, Any
import math
from dataclasses import dataclass, asdict


def validate_finite_float(val: Any, field_name: str) -> float:
    """Ensures value is numeric, not null/None, and finite (not NaN or Inf)."""
    if val is None or not isinstance(val, (int, float)) or isinstance(val, bool):
        raise ValueError(f"Field '{field_name}' must be a finite float, got {val!r}")
    f_val = float(val)
    if not math.isfinite(f_val):
        raise ValueError(f"Field '{field_name}' contains non-finite value: {f_val}")
    return f_val


def validate_vec3(val: Any, field_name: str) -> List[float]:
    """Ensures value is a 3-element list of finite floats."""
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        raise ValueError(f"Field '{field_name}' must be a list of 3 finite floats, got {val!r}")
    return [validate_finite_float(v, f"{field_name}[{i}]") for i, v in enumerate(val)]


def validate_int_ms(val: Any, field_name: str) -> int:
    """Ensures integer monotonic millisecond timestamp or sequence."""
    if not isinstance(val, int) or isinstance(val, bool):
        raise ValueError(f"Field '{field_name}' must be an integer, got {val!r}")
    return int(val)


# ---- Inbound Models ----

@dataclass
class GPSFix:
    lat: float
    lon: float
    alt_m: float
    speed_mps: float
    bearing_deg: float
    accuracy_m: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPSFix":
        if not isinstance(data, dict):
            raise ValueError("GPS fix must be a JSON object")
        return cls(
            lat=validate_finite_float(data.get("lat"), "lat"),
            lon=validate_finite_float(data.get("lon"), "lon"),
            alt_m=validate_finite_float(data.get("alt_m", 0.0), "alt_m"),
            speed_mps=validate_finite_float(data.get("speed_mps"), "speed_mps"),
            bearing_deg=validate_finite_float(data.get("bearing_deg"), "bearing_deg"),
            accuracy_m=validate_finite_float(data.get("accuracy_m", 0.0), "accuracy_m"),
        )


@dataclass
class GPSHistorySample:
    t_ms: int
    speed_mps: float
    bearing_deg: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPSHistorySample":
        if not isinstance(data, dict):
            raise ValueError("GPS history sample must be a JSON object")
        return cls(
            t_ms=validate_int_ms(data.get("t_ms"), "t_ms"),
            speed_mps=validate_finite_float(data.get("speed_mps"), "speed_mps"),
            bearing_deg=validate_finite_float(data.get("bearing_deg"), "bearing_deg"),
        )


@dataclass
class IMUHistorySample:
    t_ms: int
    accel: List[float]
    gyro: List[float]
    gravity: List[float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMUHistorySample":
        if not isinstance(data, dict):
            raise ValueError("IMU history sample must be a JSON object")
        return cls(
            t_ms=validate_int_ms(data.get("t_ms"), "t_ms"),
            accel=validate_vec3(data.get("accel"), "accel"),
            gyro=validate_vec3(data.get("gyro"), "gyro"),
            gravity=validate_vec3(data.get("gravity"), "gravity"),
        )


@dataclass
class CalibrateRequest:
    session_id: str
    seq: int
    t_ms: int
    sample_rate_hz: float
    allow_low_health: bool
    last_gps: GPSFix
    gps_history: List[GPSHistorySample]
    imu_history: List[IMUHistorySample]

    road_polyline: Optional[List[List[float]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrateRequest":
        if not isinstance(data, dict):
            raise ValueError("Payload must be a JSON object")
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Field 'session_id' must be a non-empty string")

        seq = validate_int_ms(data.get("seq"), "seq")
        t_ms = validate_int_ms(data.get("t_ms"), "t_ms")
        sample_rate_hz = validate_finite_float(data.get("sample_rate_hz", 10.0), "sample_rate_hz")
        allow_low_health = bool(data.get("allow_low_health", False))

        last_gps_data = data.get("last_gps")
        if not last_gps_data:
            raise ValueError("Field 'last_gps' is required")
        last_gps = GPSFix.from_dict(last_gps_data)

        gps_hist_raw = data.get("gps_history")
        if not isinstance(gps_hist_raw, list):
            raise ValueError("Field 'gps_history' must be a list")
        gps_history = [GPSHistorySample.from_dict(g) for g in gps_hist_raw]

        imu_hist_raw = data.get("imu_history")
        if not isinstance(imu_hist_raw, list):
            raise ValueError("Field 'imu_history' must be a list")
        imu_history = [IMUHistorySample.from_dict(im) for im in imu_hist_raw]

        road_polyline = data.get("road_polyline")

        return cls(
            session_id=session_id,
            seq=seq,
            t_ms=t_ms,
            sample_rate_hz=sample_rate_hz,
            allow_low_health=allow_low_health,
            last_gps=last_gps,
            gps_history=gps_history,
            imu_history=imu_history,
            road_polyline=road_polyline,
        )


@dataclass
class SensorPacket:
    session_id: str
    seq: int
    t_ms: int
    accel: List[float]
    gyro: List[float]
    gravity: List[float]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorPacket":
        if not isinstance(data, dict):
            raise ValueError("Payload must be a JSON object")
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Field 'session_id' must be a non-empty string")
        seq = validate_int_ms(data.get("seq"), "seq")
        t_ms = validate_int_ms(data.get("t_ms"), "t_ms")
        accel = validate_vec3(data.get("accel"), "accel")
        gyro = validate_vec3(data.get("gyro"), "gyro")
        gravity = validate_vec3(data.get("gravity"), "gravity")
        return cls(
            session_id=session_id,
            seq=seq,
            t_ms=t_ms,
            accel=accel,
            gyro=gyro,
            gravity=gravity,
        )


@dataclass
class GPSRestoredPacket:
    session_id: str
    seq: int
    t_ms: int
    gps: GPSFix

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPSRestoredPacket":
        if not isinstance(data, dict):
            raise ValueError("Payload must be a JSON object")
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Field 'session_id' must be a non-empty string")
        seq = validate_int_ms(data.get("seq"), "seq")
        t_ms = validate_int_ms(data.get("t_ms"), "t_ms")
        gps_raw = data.get("gps")
        if not gps_raw:
            raise ValueError("Field 'gps' is required")
        gps = GPSFix.from_dict(gps_raw)
        return cls(session_id=session_id, seq=seq, t_ms=t_ms, gps=gps)


@dataclass
class ResetPacket:
    session_id: str
    seq: int
    t_ms: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResetPacket":
        if not isinstance(data, dict):
            raise ValueError("Payload must be a JSON object")
        session_id = data.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Field 'session_id' must be a non-empty string")
        seq = validate_int_ms(data.get("seq"), "seq")
        t_ms = validate_int_ms(data.get("t_ms"), "t_ms")
        return cls(session_id=session_id, seq=seq, t_ms=t_ms)


# ---- Outbound Models ----

@dataclass
class CalibrateDoneResponse:
    session_id: str
    seq: int
    t_ms: int
    status: str
    degraded: bool
    r_phone_to_vehicle: List[List[float]]
    alignment_health: float
    fwd_accel_bias_mps2: float
    gyro_bias_radps: List[float]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PositionPayload:
    session_id: str
    seq: int
    t_ms: int
    lat: float
    lon: float
    alt_m: float
    enu_x_m: float
    enu_y_m: float
    enu_z_m: float
    speed_mps: float
    bearing_deg: float
    mode: str
    degraded: bool
    confidence: float
    is_stopped: bool
    shock_detected: bool
    stale: bool
    gap_s: float
    kinonet_speed_mps: float
    kinonet_var: float
    inekf_e_m: float
    inekf_n_m: float
    dr_lat: Optional[float]
    dr_lon: Optional[float]
    dr_error_m: Optional[float]
    dr_distance_m: float
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResetDoneResponse:
    session_id: str
    seq: int
    t_ms: int
    status: str
    mode: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorResponse:
    session_id: str
    seq: int
    t_ms: int
    code: str
    message: str
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
