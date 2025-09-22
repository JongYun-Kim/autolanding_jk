# pip install numpy opencv-python shapely
import numpy as np
import cv2
from shapely.geometry import Polygon, box, LineString
from shapely.validation import explain_validity


class CameraVisibilityChecker:
    """
    Robust visibility/coverage check for a planar 3D quad in a pinhole camera.

    Key differences vs the original:
      - Clips the polygon in CAMERA SPACE against the near plane and frustum
        (x in [-hx*z, +hx*z], y in [-hy*z, +hy*z], z >= z_near) before projection.
      - Projects the CLIPPED polygon and re-orders vertices CCW in 2D to avoid bow-ties.
      - Uses intrinsics synthesized from FOV to normalize image to [-1,1]^2.
    """

    def __init__(self,
                 fov_deg: float,
                 aspect: float = 1.0,
                 z_near: float = 4e-5,       # slightly less tiny for numerical sanity
                 z_far: float = 1e9,
                 eps_planar: float = 1e-6,
                 eps_geom: float = 1e-9,
                 eps_buffer: float = 1e-6,
                 fov_is_vertical: bool = True):
        if aspect <= 0 or not np.isfinite(aspect):
            raise ValueError(f"aspect must be positive finite, got {aspect}")
        self.aspect = float(aspect)

        self.fov_deg = float(fov_deg)
        self.is_fov_deg_vertical = bool(fov_is_vertical)

        fov = np.deg2rad(self.fov_deg)
        half_min = 1e-6
        half_max = np.pi/2 - 1e-6

        if fov_is_vertical:
            fovy_half = np.clip(0.5 * fov, half_min, half_max)
            fy = 1.0 / np.tan(fovy_half)
            fx = fy / self.aspect
            self.fovy = 2.0 * np.arctan(1.0 / fy)
            self.fovx = 2.0 * np.arctan(1.0 / fx)
        else:
            fovx_half = np.clip(0.5 * fov, half_min, half_max)
            fx = 1.0 / np.tan(fovx_half)
            fy = fx * self.aspect
            self.fovx = 2.0 * np.arctan(1.0 / fx)
            self.fovy = 2.0 * np.arctan(1.0 / fy)

        # convenient half-angles in tangent form for frustum planes
        self.hx = np.tan(self.fovx * 0.5)
        self.hy = np.tan(self.fovy * 0.5)

        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.eps_planar = float(eps_planar)
        self.eps_geom = float(eps_geom)
        self.eps_buffer = float(eps_buffer)

        # camera intrinsics that map to normalized [-1,1]^2
        self.K = np.array([[1.0 / self.hx, 0.0, 0.0],
                           [0.0, 1.0 / self.hy, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.distCoeffs = np.zeros((4, 1), dtype=np.float64)  # zero-distortion

        # normalized image frame as shapely polygon
        self.img_poly = box(-1.0, -1.0, 1.0, 1.0)
        self.img_area = float(self.img_poly.area)  # == 4.0

        # debug/outputs
        self.target_coverage = None
        self._proj = None      # projected 2D poly (Nx2)
        self._Pc_clip = None   # clipped camera-space poly (Nx3)
        self._Pc_raw = None    # raw camera-space quad (4x3)

    @staticmethod
    def _normalize(v, fallback=None):
        n = np.linalg.norm(v)
        if n < 1e-12:
            if fallback is not None:
                return fallback
            raise ValueError("Zero-length vector encountered.")
        return v / n

    @staticmethod
    def _order_quad_on_plane(rect_xyz, eps_planar=1e-6):
        """
        Ensure 4 vertices are coplanar (within tolerance) and return them in CCW order on that plane.
        Works even if input order is scrambled.
        """
        if rect_xyz.shape != (4, 3):
            raise ValueError("rect_xyz must be (4,3).")

        P = rect_xyz
        # Find a triplet with non-degenerate area
        idxs = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
        normal = None
        base = None
        for i,j,k in idxs:
            v1 = P[j] - P[i]
            v2 = P[k] - P[i]
            n = np.cross(v1, v2)
            if np.linalg.norm(n) > eps_planar:
                normal = n
                base = P[i]
                break
        if normal is None:
            raise ValueError("All four points are nearly collinear; not a valid rectangle.")
        n_hat = CameraVisibilityChecker._normalize(normal)

        dists = np.abs((P - base) @ n_hat)
        if np.max(dists) > eps_planar:
            raise ValueError(f"Input points are not coplanar: max dist = {np.max(dists):.3e}")

        edges = [P[(i + 1) % 4] - P[i] for i in range(4)]
        e1 = edges[np.argmax([np.linalg.norm(e) for e in edges])]
        e1 = CameraVisibilityChecker._normalize(e1)
        e2 = CameraVisibilityChecker._normalize(np.cross(n_hat, e1))

        ST = np.stack([((p - base) @ e1, (p - base) @ e2) for p in P], axis=0)

        c = np.mean(ST, axis=0)
        angles = np.arctan2(ST[:, 1] - c[1], ST[:, 0] - c[0])
        order = np.argsort(angles)
        P_ordered = P[order]
        return P_ordered, n_hat

    @staticmethod
    def _make_extrinsics(cam_pos, cam_forward, cam_up):
        """
        Build R (world->cam) and t for OpenCV-like convention: +Z_cam is forward.
        Returns (rvec, tvec, R_wc).
        """
        f = CameraVisibilityChecker._normalize(cam_forward)
        up = CameraVisibilityChecker._normalize(cam_up)

        # up과 f가 거의 평행이면 대체 up 사용
        if np.linalg.norm(np.cross(up, f)) < 1e-8:
            alt = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = CameraVisibilityChecker._normalize(alt)

        # Right-handed basis!!
        r = CameraVisibilityChecker._normalize(np.cross(up, f))  # right = up × f
        u = CameraVisibilityChecker._normalize(np.cross(f, r))  # true up = f × r

        R_wc = np.stack([r, u, f], axis=0)  # world->cam
        # 수치 문제 대비: det이 음수면 r 뒤집어 교정
        if np.linalg.det(R_wc) < 0:
            r = -r
            R_wc = np.stack([r, u, f], axis=0)
        # R_wc가 proper rotation인지 확인
        assert np.isclose(np.linalg.det(R_wc), 1.0, atol=1e-6), "R_wc is not a proper rotation (det != +1)."

        t = -R_wc @ cam_pos.reshape(3,)
        rvec, _ = cv2.Rodrigues(R_wc.astype(np.float64))
        tvec = t.reshape(3, 1).astype(np.float64)
        return rvec, tvec, R_wc

    @staticmethod
    def _world_to_cam(Pw, R_wc, cam_pos):
        """Compute camera-frame coordinates directly (for Z and frustum tests)."""
        return (R_wc @ (Pw - cam_pos[None, :]).T).T  # (N,3)

    # ---- clipping helpers ----
    def _clip_halfspace(self, poly, inside_fn, intersect_t_fn):
        """
        Generic Sutherland–Hodgman step for a 3D polygon in camera space.
        poly: list of 3D points (np.array(3,))
        inside_fn: p -> bool
        intersect_t_fn: (A,B) -> t in [0,1] or None
        """
        if not poly:
            return []
        out = []
        n = len(poly)
        for i in range(n):
            A = poly[i]
            B = poly[(i + 1) % n]
            Ain = inside_fn(A)
            Bin = inside_fn(B)
            if Ain and Bin:
                out.append(B)
            elif Ain and not Bin:
                t = intersect_t_fn(A, B)
                if t is not None and 0.0 <= t <= 1.0:
                    out.append(A + t * (B - A))
            elif (not Ain) and Bin:
                t = intersect_t_fn(A, B)
                if t is not None and 0.0 <= t <= 1.0:
                    out.append(A + t * (B - A))
                out.append(B)
            # else: both out → add nothing
        return out

    def _clip_to_frustum(self, Pc):
        """
        Clip a camera-space polygon against:
            z >= z_near,
            x <=  hx*z, x >= -hx*z,
            y <=  hy*z, y >= -hy*z.
        Returns a list of 3D points (possibly empty).
        """
        Pc = [np.asarray(p, dtype=float) for p in Pc]
        hx, hy = self.hx, self.hy
        znear = self.z_near

        # z >= znear
        def in_near(p): return p[2] >= znear
        def t_near(A, B):
            dz = B[2] - A[2]
            if abs(dz) < 1e-12: return None
            return (znear - A[2]) / dz

        poly = self._clip_halfspace(Pc, in_near, t_near)
        if len(poly) < 3:
            return []

        # x <= hx*z
        def in_right(p): return p[0] <= hx * p[2]
        def t_right(A, B):
            dx, dz = B[0] - A[0], B[2] - A[2]
            denom = dx - hx * dz
            if abs(denom) < 1e-12: return None
            return (hx * A[2] - A[0]) / denom

        poly = self._clip_halfspace(poly, in_right, t_right)
        if len(poly) < 3:
            return []

        # x >= -hx*z
        def in_left(p): return p[0] >= -hx * p[2]
        def t_left(A, B):
            dx, dz = B[0] - A[0], B[2] - A[2]
            denom = dx + hx * dz
            if abs(denom) < 1e-12: return None
            return (-hx * A[2] - A[0]) / denom

        poly = self._clip_halfspace(poly, in_left, t_left)
        if len(poly) < 3:
            return []

        # y <= hy*z
        def in_top(p): return p[1] <= hy * p[2]
        def t_top(A, B):
            dy, dz = B[1] - A[1], B[2] - A[2]
            denom = dy - hy * dz
            if abs(denom) < 1e-12: return None
            return (hy * A[2] - A[1]) / denom

        poly = self._clip_halfspace(poly, in_top, t_top)
        if len(poly) < 3:
            return []

        # y >= -hy*z
        def in_bottom(p): return p[1] >= -hy * p[2]
        def t_bottom(A, B):
            dy, dz = B[1] - A[1], B[2] - A[2]
            denom = dy + hy * dz
            if abs(denom) < 1e-12: return None
            return (-hy * A[2] - A[1]) / denom

        poly = self._clip_halfspace(poly, in_bottom, t_bottom)
        return poly

    def is_visible(self, rect_xyz, cam_pos, cam_forward, cam_up, min_fraction: float = 0.0):
        """
        Returns True if the rectangle covers at least `min_fraction` of the frame area.
        - min_fraction: fraction of normalized frame area ([-1,1]^2 has area 4).
          If 0, line-only intersections also count as "visible".
        """
        self._proj = None
        self._Pc_clip = None
        self._Pc_raw = None
        self.target_coverage = -1.0

        rect_xyz = np.asarray(rect_xyz, dtype=np.float64).reshape(4, 3)
        cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3,)
        cam_forward = np.asarray(cam_forward, dtype=np.float64).reshape(3,)
        cam_up = np.asarray(cam_up, dtype=np.float64).reshape(3,)

        # 1) Validate & order quad on plane (world space)
        rect_ordered, _ = self._order_quad_on_plane(rect_xyz, self.eps_planar)

        # 2) Extrinsics & camera-space coordinates
        rvec, tvec, R_wc = self._make_extrinsics(cam_pos, cam_forward, cam_up)
        Pc = self._world_to_cam(rect_ordered, R_wc, cam_pos)  # (4,3), raw
        self._Pc_raw = Pc.copy()

        # Quick reject: fully behind or beyond far
        Z = Pc[:, 2]
        if np.all(Z < self.z_near) or np.all(Z > self.z_far):
            return False

        # 3) Clip polygon in camera space against near & frustum
        Pc_clip = self._clip_to_frustum(Pc.tolist())
        if len(Pc_clip) < 3:
            return False
        Pc_clip = np.asarray(Pc_clip, dtype=np.float64)
        self._Pc_clip = Pc_clip.copy()

        # 4) Project clipped polygon to normalized image plane:
        #    uv = (x/z, y/z), then scale by fx, fy so edges map to +/-1
        uv = Pc_clip[:, :2] / Pc_clip[:, 2:3]
        proj = (self.K[:2, :2] @ uv.T).T  # (N,2) in [-1,1] ideally
        self._proj = proj.copy()

        # 5) Robust 2D CCW ordering to avoid bow-ties from numeric jitter
        c = proj.mean(axis=0)
        ang = np.arctan2(proj[:, 1] - c[1], proj[:, 0] - c[0])
        proj = proj[np.argsort(ang)]

        # 6) Shapely polygon & coverage
        poly2d = Polygon(proj)
        if not poly2d.is_valid:
            # Avoid masking real issues silently; still try tiny buffer
            poly2d = poly2d.buffer(self.eps_buffer)

        # Check if the target is big enough to have visibility
        if poly2d.is_valid and poly2d.area >= self.eps_geom:
            inter = poly2d.intersection(self.img_poly)
            # When empty, no visibility
            if inter.is_empty:
                return False

            # if area > 0, let's see
            area = getattr(inter, "area", 0.0)
            self.target_coverage = float(area / self.img_area) if area > 0.0 else 0.0

            if min_fraction > 0.0:
                return self.target_coverage >= float(min_fraction)
            else:
                if area > 0.0:
                    return True
                length = getattr(inter, "length", 0.0)  # degenerated!
                return (length is not None) and (length > 0.0)

        # Polygon collapsed to line/point after clipping (extremely rare if min_fraction>0)
        if min_fraction > 0.0:
            return False

        # Treat line intersection as visible
        try:
            coords = np.array(poly2d.exterior.coords) if poly2d.exterior else np.array(self._proj)
            if coords.shape[0] >= 2:
                line = LineString(coords)
                return line.buffer(self.eps_buffer).intersects(self.img_poly)
            else:
                pt = proj.mean(axis=0)
                tiny = Polygon([pt, pt + [1e-12, 0], pt + [0, 1e-12]])
                return self.img_poly.buffer(0).contains(tiny)
        except Exception:
            return False


if __name__ == "__main__":
    checker = CameraVisibilityChecker(fov_deg=90.0, aspect=1)

    rect = np.array([
        [ 0.25, -0.25,  0.0],
        [ 0.25,  0.25,  0.0],
        [-0.25,  0.25,  0.0],
        [-0.25, -0.25,  0.0],
    ])

    cam_pos = np.array([0.0, 0.0, 0.5405346])
    cam_forward = np.array([0.611436, 0.611436, -1.0])
    cam_up = np.array([1.0, 1.0, 0.0])

    visible = checker.is_visible(rect, cam_pos, cam_forward, cam_up, min_fraction=0.02)
    print("visible:", visible)
    print("coverage:", checker.target_coverage)

    print("...stop!")