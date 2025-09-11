# pip install numpy opencv-python shapely
import numpy as np
import cv2
from shapely.geometry import Polygon, box, LineString
from shapely.validation import explain_validity

class CameraVisibilityChecker:
    """
    Check if any part of a 3D rectangle is visible in a pinhole camera with zero distortion.

    Inputs (world frame):
      - rect_xyz: (4,3) ndarray, rectangle vertices (unordered ok).
      - cam_pos: (3,) ndarray
      - cam_forward: (3,) ndarray
      - cam_up: (3,) ndarray  (not parallel to forward)
      - fov_deg: float (*VERTICAL* FOV in degrees if fov_is_vertical=True, else *HORIZONTAL* FOV), default 90.0)
      - aspect: float (*WIDTH/HEIGHT*), default 1.0
    Assumptions:
      - Pinhole camera, no distortion.
      - OpenCV convention for projection; we synthesize intrinsics from FOV.
      - The image frame is the normalized square [-1,1]x[-1,1] **after** using fx=1/tan(fovx/2), fy=1/tan(fovy/2).
    """

    def __init__(self,
                 fov_deg: float,
                 aspect: float = 1.0,
                 z_near: float = 1e-6, z_far: float = 1e9,
                 eps_planar: float = 1e-6, eps_geom: float = 1e-9, eps_buffer: float = 1e-6,
                 fov_is_vertical: bool = True
                 ):

        if aspect <= 0 or not np.isfinite(aspect):
            raise ValueError(f"aspect must be positive finite, got {aspect}")
        self.aspect = float(aspect)

        fov = np.deg2rad(float(fov_deg))
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

        self.z_near = float(z_near)
        self.z_far = float(z_far)
        self.eps_planar = float(eps_planar)
        self.eps_geom = float(eps_geom)
        self.eps_buffer = float(eps_buffer)

        # camera intrinsics for normalized image box [-1,1]^2
        self.K = np.array([[fx, 0.0, 0.0],
                           [0.0, fy, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        self.distCoeffs = np.zeros((4, 1), dtype=np.float64)  # zero-distortion

        # normalized image frame as shapely polygon
        self.img_poly = box(-1.0, -1.0, 1.0, 1.0)
        self.img_area = float(self.img_poly.area)  # == 4.0

        self.target_coverage = None

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

        # Compute plane from first three non-collinear points
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

        # Planarity check: distance of each point to the plane
        dists = np.abs((P - base) @ n_hat)
        if np.max(dists) > eps_planar:
            raise ValueError(f"Input points are not coplanar within tolerance: max dist = {np.max(dists):.3e}")

        # Build 2D coordinates on the plane for ordering
        # Choose e1 along the longest edge direction for stability
        edges = [P[(i+1)%4] - P[i] for i in range(4)]
        e1 = edges[np.argmax([np.linalg.norm(e) for e in edges])]
        e1 = CameraVisibilityChecker._normalize(e1)
        e2 = CameraVisibilityChecker._normalize(np.cross(n_hat, e1))

        # Project to (s,t)
        ST = np.stack([((p - base) @ e1, (p - base) @ e2) for p in P], axis=0)

        # Order points CCW around centroid
        c = np.mean(ST, axis=0)
        angles = np.arctan2(ST[:,1] - c[1], ST[:,0] - c[0])
        order = np.argsort(angles)
        P_ordered = P[order]
        return P_ordered, n_hat

    @staticmethod
    def _make_extrinsics(cam_pos, cam_forward, cam_up):
        """
        Build R (world->cam) and t for OpenCV projectPoints so that +Z_cam is "forward".
        """
        f = CameraVisibilityChecker._normalize(cam_forward)
        up = CameraVisibilityChecker._normalize(cam_up)
        # Fix near-parallel up
        if np.linalg.norm(np.cross(f, up)) < 1e-8:
            # pick an alternate up
            alt = np.array([0.0, 1.0, 0.0]) if abs(f[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            up = CameraVisibilityChecker._normalize(alt)

        r = CameraVisibilityChecker._normalize(np.cross(f, up))   # right
        u = np.cross(r, f)                                        # true up, orthonormal

        # world->cam rotation: rows are cam axes in world basis
        R_wc = np.stack([r, u, f], axis=0)  # 3x3
        # OpenCV expects rvec/tvec such that X_cam = R * (X_world - C)
        t = -R_wc @ cam_pos.reshape(3,)

        # Rodrigues vector
        rvec, _ = cv2.Rodrigues(R_wc.astype(np.float64))
        tvec = t.reshape(3,1).astype(np.float64)
        return rvec, tvec, R_wc

    def _project_points(self, Pw, rvec, tvec):
        """Project Nx3 points with our K and zero distortion; returns Nx2 array and Z_cam (depths)."""
        Pw = Pw.astype(np.float64).reshape(-1, 1, 3)
        img_pts, _ = cv2.projectPoints(Pw, rvec, tvec, self.K, self.distCoeffs)
        img_pts = img_pts.reshape(-1, 2)
        # Recover Z_cam to apply near/far clipping and handle behind-camera points
        # X_cam = R*(X - C) = we can approximate via small finite differencing OR pass R_wc from caller.
        # Simpler: compute explicitly using R,t like OpenCV does:
        # -> We'll compute X_cam directly outside and pass into this when needed.
        return img_pts

    @staticmethod
    def _world_to_cam(Pw, R_wc, cam_pos):
        """Compute camera-frame coordinates directly (for Z and frustum tests)."""
        return (R_wc @ (Pw - cam_pos[None, :]).T).T  # (N,3)

    def is_visible(self, rect_xyz, cam_pos, cam_forward, cam_up, min_fraction: float = 0.0):
        """
        Returns True if the rectangle covers at least `min_fraction` of the frame area.
        - min_fraction: 프레임([-1,1]^2) 대비 최소 면적 비율(예: 0.01은 1%).
          0일 때는 기존과 동일하게 '살짝 걸쳐도 보임'(선분 교차 포함)을 허용.
        """
        self.target_coverage = -1

        rect_xyz = np.asarray(rect_xyz, dtype=np.float64).reshape(4,3)
        cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3,)
        cam_forward = np.asarray(cam_forward, dtype=np.float64).reshape(3,)
        cam_up = np.asarray(cam_up, dtype=np.float64).reshape(3,)

        # 1) Validate and order the rectangle on its plane
        rect_ordered, _ = self._order_quad_on_plane(rect_xyz, self.eps_planar)

        # 2) Build extrinsics and depths
        rvec, tvec, R_wc = self._make_extrinsics(cam_pos, cam_forward, cam_up)
        Pc = self._world_to_cam(rect_ordered, R_wc, cam_pos)  # for Z
        Z = Pc[:,2]

        # Quick reject: all behind or beyond far
        if np.all(Z <= self.z_near) or np.all(Z >= self.z_far):
            return False

        # 3) Project (OpenCV)
        proj = self._project_points(rect_ordered, rvec, tvec)  # shape (4,2)

        # 4) 2D 폴리곤 구성 및 수치 안정화
        poly2d = Polygon(proj)
        if not poly2d.is_valid:
            # Try fixing tiny self-intersections caused by numerical noise
            poly2d = poly2d.buffer(self.eps_buffer)

        # 5) 면적 기반 판정 경로 (사각형이 폴리곤으로 유지되는 경우)
        if poly2d.is_valid and poly2d.area >= self.eps_geom:
            inter = poly2d.intersection(self.img_poly)
            if inter.is_empty:
                return False

            # 면적 교집합이 존재하면 커버리지 계산
            area = getattr(inter, "area", 0.0)
            self.target_coverage = float(area / self.img_area) if area > 0.0 else 0.0

            if min_fraction > 0.0:
                # 최소 면적 기준 적용
                return self.target_coverage >= float(min_fraction)
            else:
                # 기존 동작: 면적>0 이면 True, 아니면 길이(선분 교차)도 허용
                if area > 0.0:
                    return True
                length = getattr(inter, "length", 0.0)
                return (length is not None) and (length > 0.0)

        # 6) 폴리곤이 선/점으로 붕괴된 경우 (근평면/왜곡 등)
        #    min_fraction>0이면 면적 기준을 만족할 수 없으므로 바로 False.
        if min_fraction > 0.0:
            return False

        # 기존 동작 유지: 선분 교차만 있어도 '보임'으로 간주
        # poly2d.exterior가 없을 수 있으므로 좌표로 LineString 생성
        try:
            coords = np.array(poly2d.exterior.coords) if poly2d.exterior else np.array(proj)
            if coords.shape[0] >= 2:
                line = LineString(coords)
                return line.buffer(self.eps_buffer).intersects(self.img_poly)
            else:
                # 점 교차: 아주 미세한 버퍼로 포함 여부 판정
                pt = proj.mean(axis=0)
                tiny = Polygon([pt, pt + [1e-12, 0], pt + [0, 1e-12]])
                return self.img_poly.buffer(0).contains(tiny)
        except Exception:
            return False


if __name__ == "__main__":
    checker = CameraVisibilityChecker(fov_deg=90.0, aspect=1)

    rect = np.array([
        [ 1.0, -1.0,  0.0],
        [ 1.0,  1.0,  0.0],
        [-1.0,  1.0,  0.0],
        [-1.0, -1.0,  0.0],
    ])
    # rect = np.array([
    #     [-1.0, -1.0,  0.0],
    #     [-1.0,  1.0,  0.0],
    #     [ 1.0,  1.0,  0.0],
    #     [ 1.0, -1.0,  0.0],
    # ])
    cam_pos = np.array([0.0, 0.0, 2.0])
    cam_forward = np.array([0.0, 0.0, -1.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    visible = checker.is_visible(rect, cam_pos, cam_forward, cam_up, min_fraction=0.1)
    print("visible:", visible)
    print("coverage:", checker.target_coverage)

    print("...stop!")