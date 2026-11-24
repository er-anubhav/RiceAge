"""
Dual-View Scanner Platen Registration and Health Checks.
Applies Phase 3.2 raw sensor coordinate conventions, verifies geometric validity,
and provides fail-closed safeguards against uncalibrated or distorted homographies.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


class DualViewRegistration:
    """
    Manages coordinate mapping from bottom platen to top platen.
    Enforces reflection chirality, scale conservation, and fail-closed safety.
    """

    # Baseline direct homography in 600 DPI raw coordinates
    DEFAULT_HOMOGRAPHY = np.array([
        [-0.998463, -0.000463, 4917.872],
        [ 0.001606,  0.999657,   21.086],
        [ 0.000000,  0.000000,    1.000]
    ], dtype=np.float64)

    @classmethod
    def get_default_homography(cls, dpi_bottom: int = 600, dpi_top: int = 600) -> np.ndarray:
        """Scales baseline 600 DPI homography to match scanner acquisition DPIs."""
        s_b = float(dpi_bottom) / 600.0
        s_t = float(dpi_top) / 600.0
        H = cls.DEFAULT_HOMOGRAPHY.copy()
        H[0, 0] = H[0, 0] * (s_t / s_b)
        H[0, 1] = H[0, 1] * (s_t / s_b)
        H[0, 2] = H[0, 2] * s_t
        H[1, 0] = H[1, 0] * (s_t / s_b)
        H[1, 1] = H[1, 1] * (s_t / s_b)
        H[1, 2] = H[1, 2] * s_t
        return H

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        dpi_bottom: int = 600,
        dpi_top: int = 600,
        min_determinant: float = -1.05,
        max_determinant: float = -0.95,
        min_scale: float = 0.95,
        max_scale: float = 1.05,
        max_reprojection_rmse_px: float = 8.0,
        sensor_width_px: Optional[int] = None,
        sensor_height_px: Optional[int] = None
    ):
        self.dpi_bottom = dpi_bottom
        self.dpi_top = dpi_top
        self.scale_ratio = float(dpi_top) / float(dpi_bottom) if dpi_bottom > 0 else 1.0

        if homography_matrix is not None:
            self.H = np.array(homography_matrix, dtype=np.float64)
        else:
            self.H = self.get_default_homography(dpi_bottom=dpi_bottom, dpi_top=dpi_top)

        scale_top = float(dpi_top) / 600.0
        self.min_det = min_determinant
        self.max_det = max_determinant
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.max_reprojection_rmse_px = max_reprojection_rmse_px * scale_top
        self.sensor_w = sensor_width_px if sensor_width_px is not None else int(round(4928 * scale_top))
        self.sensor_h = sensor_height_px if sensor_height_px is not None else int(round(7000 * scale_top))

        self._health_status: Dict[str, Any] = {}
        self.is_healthy = self._verify_calibration_health()

    def _verify_calibration_health(self) -> bool:
        """
        Runs mathematical sanity checks on the transformation matrix.
        Returns True if matrix passes all optical/physical invariants.
        """
        if self.H.shape != (3, 3):
            self._health_status = {"valid": False, "reason": f"Invalid shape: {self.H.shape}"}
            return False

        if not np.isfinite(self.H).all():
            self._health_status = {"valid": False, "reason": "Non-finite elements in matrix"}
            return False

        # Affine sub-matrix normalized by cross-DPI scale ratio
        r = getattr(self, "scale_ratio", 1.0)
        A_norm = self.H[:2, :2] / max(r, 1e-6)
        det = float(np.linalg.det(A_norm))

        # Normalized scale factors
        sx = float(np.sqrt(A_norm[0, 0]**2 + A_norm[1, 0]**2))
        sy = float(np.sqrt(A_norm[0, 1]**2 + A_norm[1, 1]**2))

        # In-plane rotation
        theta_rad = float(np.arctan2(self.H[1, 0], self.H[0, 0]))
        theta_deg = float(np.degrees(theta_rad))

        tx = float(self.H[0, 2])
        ty = float(self.H[1, 2])

        is_valid_det = (self.min_det <= det <= self.max_det)
        is_valid_sx = (self.min_scale <= sx <= self.max_scale)
        is_valid_sy = (self.min_scale <= sy <= self.max_scale)

        is_healthy = is_valid_det and is_valid_sx and is_valid_sy

        reason = "OK" if is_healthy else []
        if not is_valid_det:
            reason.append(f"Determinant {det:.4f} outside bounds [{self.min_det}, {self.max_det}]")
        if not is_valid_sx:
            reason.append(f"Sx {sx:.4f} outside bounds [{self.min_scale}, {self.max_scale}]")
        if not is_valid_sy:
            reason.append(f"Sy {sy:.4f} outside bounds [{self.min_scale}, {self.max_scale}]")
        if isinstance(reason, list):
            reason = "; ".join(reason)

        self._health_status = {
            "valid": is_healthy,
            "determinant": det,
            "sx": sx,
            "sy": sy,
            "theta_deg": theta_deg,
            "tx_px": tx,
            "ty_px": ty,
            "reason": reason
        }
        return is_healthy

    def get_health_report(self) -> Dict[str, Any]:
        """Returns registration diagnostic health status."""
        return dict(self._health_status)

    def project_bottom_to_top(self, x_b: float, y_b: float) -> Tuple[bool, float, float]:
        """
        Projects a single bottom platen coordinate into top platen space.
        Returns: (success: bool, x_t_proj: float, y_t_proj: float)
        Fails closed if registration health check failed.
        """
        if not self.is_healthy:
            return False, 0.0, 0.0

        pt = np.array([x_b, y_b, 1.0], dtype=np.float64)
        proj = self.H @ pt
        w = proj[2]
        if abs(w) < 1e-6:
            return False, 0.0, 0.0

        x_t = float(proj[0] / w)
        y_t = float(proj[1] / w)

        # Boundary sanity check (must lie within reasonable proximity of scanner glass)
        margin = 200.0
        if -margin <= x_t <= (self.sensor_w + margin) and -margin <= y_t <= (self.sensor_h + margin):
            return True, x_t, y_t
        else:
            return False, x_t, y_t

    def project_batch_bottom_to_top(self, pts_b: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Projects an Nx2 array of bottom coordinates to top platen space.
        Returns: (success: bool, pts_t_proj: np.ndarray Nx2)
        """
        if not self.is_healthy or len(pts_b) == 0:
            return False, np.zeros_like(pts_b)

        pts_homo = np.hstack([pts_b, np.ones((len(pts_b), 1), dtype=np.float64)])
        proj = (self.H @ pts_homo.T).T
        w = proj[:, 2:3]
        valid_w = (np.abs(w) > 1e-6).flatten()
        if not valid_w.all():
            return False, np.zeros_like(pts_b)

        pts_t = proj[:, :2] / w
        return True, pts_t
