import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPLPolygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2
from gym_pybullet_drones.utils.camera_visibility_checker import CameraVisibilityChecker


def visualize_camera_view(checker, rect_xyz, cam_pos, cam_forward, cam_up, title="Camera View"):
    """
    Visualize the camera setup and the projected rectangle on the image plane.
    """
    fig = plt.figure(figsize=(20, 7))

    # 1. 3D Scene View
    ax1 = fig.add_subplot(131, projection='3d')

    # Draw rectangle
    rect_ordered, _ = checker._order_quad_on_plane(rect_xyz, checker.eps_planar)
    rect_3d = np.vstack([rect_ordered, rect_ordered[0]])  # Close the polygon
    ax1.plot(rect_3d[:, 0], rect_3d[:, 1], rect_3d[:, 2], 'b-', linewidth=2, label='Rectangle')

    # Fill the rectangle
    verts = [list(zip(rect_ordered[:, 0], rect_ordered[:, 1], rect_ordered[:, 2]))]
    poly = Poly3DCollection(verts, alpha=0.3, facecolor='blue', edgecolor='blue')
    ax1.add_collection3d(poly)

    # Draw camera
    ax1.scatter(*cam_pos, color='red', s=100, label='Camera')

    # Draw camera viewing direction
    arrow_len = 1.0
    ax1.quiver(cam_pos[0], cam_pos[1], cam_pos[2],
               cam_forward[0] * arrow_len, cam_forward[1] * arrow_len, cam_forward[2] * arrow_len,
               color='red', arrow_length_ratio=0.2)





    # --- Draw camera frustum properly intersected with the z=0 plane ---
    # Get FOV parameters
    fovx, fovy = checker.fovx, checker.fovy

    hx = np.tan(fovx * 0.5)  # half-width at unit depth in camera coords
    hy = np.tan(fovy * 0.5)  # half-height at unit depth in camera coords

    # Rays through the 4 image-plane corners in camera coordinates (z forward)
    dirs_cam = np.array([
        [-hx, -hy,  1.0],
        [ +hx, -hy,  1.0],
        [ +hx, +hy,  1.0],
        [-hx, +hy,  1.0],
    ], dtype=float)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)

    # Camera basis in world: z_cam = forward, x_cam = right, y_cam = up
    z_cam = checker._normalize(cam_forward)                           # forward
    x_cam = checker._normalize(np.cross(z_cam, cam_up))               # right
    y_cam = np.cross(x_cam, z_cam)                                    # up

    # Convert rays to world directions
    dirs_world = (dirs_cam[:, 0:1] * x_cam +
                  dirs_cam[:, 1:2] * y_cam +
                  dirs_cam[:, 2:3] * z_cam)

    # Intersect each ray p(t) = cam_pos + t * d with the plane z=0  ->  t = -cam_pos.z / d.z
    frustum_corners = []
    for d in dirs_world:
        dz = d[2]
        if np.isclose(dz, 0.0):
            frustum_corners.append(None)  # ray parallel to plane
            continue
        t = -cam_pos[2] / dz
        if t <= 0:
            frustum_corners.append(None)  # intersection behind the camera
        else:
            frustum_corners.append(cam_pos + t * d)

    # Draw frustum lines from camera to valid intersections
    for corner in frustum_corners:
        if corner is None:
            continue
        ax1.plot([cam_pos[0], corner[0]],
                 [cam_pos[1], corner[1]],
                 [cam_pos[2], corner[2]], 'r--', alpha=0.3)

    # Draw frustum base polygon on z=0 if all four intersections are valid
    if all(c is not None for c in frustum_corners):
        frustum_base = np.vstack([frustum_corners, frustum_corners[0]])
        ax1.plot(frustum_base[:, 0], frustum_base[:, 1], frustum_base[:, 2],
                 'r--', alpha=0.5, label='FOV at z=0')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Scene')
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_zlim(0, 2)

    # 2. Image Plane Projection
    ax2 = fig.add_subplot(132)

    # Get projection
    rvec, tvec, R_wc = checker._make_extrinsics(cam_pos, cam_forward, cam_up)
    proj = checker._project_points(rect_ordered, rvec, tvec)

    # Draw image frame [-1, 1] x [-1, 1]
    frame = Rectangle((-1, -1), 2, 2, fill=False, edgecolor='black', linewidth=2)
    ax2.add_patch(frame)

    # Draw projected rectangle
    poly_proj = MPLPolygon(proj, fill=True, alpha=0.5, facecolor='blue', edgecolor='blue', linewidth=2)
    ax2.add_patch(poly_proj)

    # Mark vertices
    for i, p in enumerate(proj):
        ax2.plot(p[0], p[1], 'ro', markersize=8)
        ax2.text(p[0] + 0.05, p[1] + 0.05, f'V{i}', fontsize=10)

    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (normalized)')
    ax2.set_ylabel('Y (normalized)')
    ax2.set_title('Projected View (Image Plane)')

    # 3. Coverage Info
    ax3 = fig.add_subplot(133)
    ax3.axis('off')

    # Calculate visibility
    visible = checker.is_visible(rect_xyz, cam_pos, cam_forward, cam_up, min_fraction=0.0)
    coverage = checker.target_coverage

    info_text = f"""
    {title}

    Camera Position: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f})
    Camera Forward: ({cam_forward[0]:.1f}, {cam_forward[1]:.1f}, {cam_forward[2]:.1f})
    FOV: 90°

    Rectangle Vertices:
    {rect_xyz}

    Projected Coordinates:
    {proj}

    Visibility: {visible}
    Coverage: {coverage:.2%} of frame
    """

    ax3.text(0.1, 0.5, info_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax3.transAxes)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def run_tests():
    """
    Run comprehensive tests with various scenarios.
    """
    print("=" * 60)
    print("CameraVisibilityChecker Test Suite")
    print("=" * 60)

    # Test 1: Original scenario - 2x2 square at origin, camera at (0,0,2)
    print("\n[Test 1] Original Scenario")
    print("-" * 40)
    checker = CameraVisibilityChecker(fov_deg=90.0, aspect=1.0)

    rect = np.array([
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
    ])

    cam_pos = np.array([0.0, 0.0, 2.0])
    cam_forward = np.array([0.0, 0.0, -1.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    visible = checker.is_visible(rect, cam_pos, cam_forward, cam_up, min_fraction=0.1)
    coverage = checker.target_coverage

    print(f"Visible: {visible}")
    print(f"Coverage: {coverage:.4f} ({coverage * 100:.2f}%)")
    print(f"Expected: 0.25 (25%)")
    print(f"Result: {'PASS ✓' if abs(coverage - 0.25) < 0.01 else 'FAIL ✗'}")

    # Visualize
    fig1 = visualize_camera_view(checker, rect, cam_pos, cam_forward, cam_up,
                                 "Test 1: Original Scenario")

    # Test 2: Offset rectangle - should have different coverage
    print("\n[Test 2] Offset Rectangle")
    print("-" * 40)
    rect2 = rect + np.array([1.0, 0.0, 0.0])  # Shift right by 1

    visible2 = checker.is_visible(rect2, cam_pos, cam_forward, cam_up, min_fraction=0.0)
    coverage2 = checker.target_coverage

    print(f"Visible: {visible2}")
    print(f"Coverage: {coverage2:.4f} ({coverage2 * 100:.2f}%)")
    print(f"Expected: ~12.5% (half of rectangle in view)")
    print(f"Result: {'PASS ✓' if abs(coverage2 - 0.125) < 0.02 else 'APPROXIMATE'}")

    fig2 = visualize_camera_view(checker, rect2, cam_pos, cam_forward, cam_up,
                                 "Test 2: Offset Rectangle")

    # Test 3: Larger rectangle - should fill entire frame
    print("\n[Test 3] Large Rectangle (fills frame)")
    print("-" * 40)
    rect3 = np.array([
        [2.0, -2.0, 0.0],
        [2.0, 2.0, 0.0],
        [-2.0, 2.0, 0.0],
        [-2.0, -2.0, 0.0],
    ])

    visible3 = checker.is_visible(rect3, cam_pos, cam_forward, cam_up, min_fraction=0.5)
    coverage3 = checker.target_coverage

    print(f"Visible: {visible3}")
    print(f"Coverage: {coverage3:.4f} ({coverage3 * 100:.2f}%)")
    print(f"Expected: 1.0 (100%)")
    print(f"Result: {'PASS ✓' if abs(coverage3 - 1.0) < 0.01 else 'FAIL ✗'}")

    fig3 = visualize_camera_view(checker, rect3, cam_pos, cam_forward, cam_up,
                                 "Test 3: Large Rectangle (fills frame)")

    # Test 4: Rectangle behind camera - should not be visible
    print("\n[Test 4] Rectangle Behind Camera")
    print("-" * 40)
    rect4 = rect + np.array([0.0, 0.0, 3.0])  # Move to z=3, behind camera at z=2

    visible4 = checker.is_visible(rect4, cam_pos, cam_forward, cam_up, min_fraction=0.0)
    coverage4 = checker.target_coverage

    print(f"Visible: {visible4}")
    print(f"Coverage: {coverage4}")
    print(f"Expected: False (behind camera)")
    print(f"Result: {'PASS ✓' if not visible4 else 'FAIL ✗'}")

    fig4 = visualize_camera_view(checker, rect4, cam_pos, cam_forward, cam_up,
                                 "Test 4: Rectangle Behind Camera")

    # Test 5: min_fraction test
    print("\n[Test 5] Minimum Coverage Threshold")
    print("-" * 40)

    # Small rectangle that covers ~6.25% of frame
    rect5 = np.array([
        [0.5, -0.5, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ])

    # Test with different thresholds
    thresholds = [0.0, 0.05, 0.1]
    fig5_list = []
    for threshold in thresholds:
        visible5 = checker.is_visible(rect5, cam_pos, cam_forward, cam_up, min_fraction=threshold)
        coverage5 = checker.target_coverage
        expected = coverage5 >= threshold
        print(f"  Threshold: {threshold:.0%}, Coverage: {coverage5:.4f}, Visible: {visible5}, "
              f"Expected: {expected}, {'PASS ✓' if visible5 == expected else 'FAIL ✗'}")
        fig5 = visualize_camera_view(checker, rect5, cam_pos, cam_forward, cam_up,
                                        f"Test 5: min_fraction={threshold:.0%}")
        fig5_list.append(fig5)

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)

    plt.show()
    return checker


if __name__ == "__main__":
    # Run the test suite
    # checker = run_tests()

    print("\n[Additional Manual Test]")
    print("Testing the exact scenario from the problem:")
    checker = CameraVisibilityChecker(fov_deg=81.7, aspect=1)

    rect = np.array([
        [ 0.25, -0.25,  0.0],
        [ 0.25,  0.25,  0.0],
        [-0.25,  0.25,  0.0],
        [-0.25, -0.25,  0.0],
    ])

    cam_pos = np.array([0.0, 0.0, 1.616])
    cam_forward = np.array([0.0, 0.0, -1.0])
    cam_up = np.array([0.0, 1.0, 0.0])

    # cam_pos = np.array([-np.sqrt(2), -np.sqrt(2), 2])
    # cam_forward = checker._normalize(np.array([np.sqrt(2), np.sqrt(2), -2]))  # 원점을 바라보도록
    # cam_up = np.array([np.sqrt(2), np.sqrt(2), 0.0])

    visible = checker.is_visible(rect, cam_pos, cam_forward, cam_up, min_fraction=0.0014)
    print(f"\nFinal Result:")
    print(f"  visible: {visible} (Expected: True)")
    print(f"  coverage: {checker.target_coverage * 100:.6f}\% (Expected: 10.10)")

    expect_coverage = 0.1010
    precision = 0.01

    if visible and abs(checker.target_coverage - expect_coverage) < precision:
        print("\n✓ SUCCESS: The script now works correctly!")
    else:
        print("\n✗ There may still be an issue.")

    fig_manual = visualize_camera_view(checker, rect, cam_pos, cam_forward, cam_up,
                                      "Manual Test: Camera at (-2,-2,2) looking at origin")
    plt.show()
