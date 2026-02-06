#!/usr/bin/env python3
"""
Test script to verify gimbal control modes work correctly.
Tests backward compatibility and all three control modes.
"""

import numpy as np
import gym
import gym_pybullet_drones

def test_position_mode():
    """Test position mode (default, backward compatible)"""
    print("\n" + "="*60)
    print("Testing POSITION mode (default, backward compatible)")
    print("="*60)

    env = gym.make('gimbal-curriculum-landing-aviary-v0')
    obs = env.reset()

    # Test a few steps with gimbal commands
    action = np.array([0.0, 0.0, 0.0, 0.5, -0.3], dtype=np.float32)  # pitch=0.5, yaw=-0.3

    for i in range(5):
        obs, reward, done, info = env.step(action)

        # In position mode, current angles should equal target immediately
        assert hasattr(env, 'gimbal_current_angles'), "gimbal_current_angles not found"
        assert hasattr(env, 'gimbal_target'), "gimbal_target not found"

        # Check that current angles match target (position mode)
        np.testing.assert_allclose(
            env.gimbal_current_angles,
            env.gimbal_target,
            rtol=1e-5,
            err_msg=f"Step {i}: Position mode should have current == target"
        )

    env.close()
    print("✓ Position mode test PASSED")
    return True

def test_velocity_mode():
    """Test velocity mode"""
    print("\n" + "="*60)
    print("Testing VELOCITY mode")
    print("="*60)

    env = gym.make('gimbal-curriculum-landing-aviary-v0',
                   gimbal_control_mode='velocity',
                   gimbal_max_velocity=3.0)
    obs = env.reset()

    # Initial position should be at initial_gimbal_target
    initial_angles = env.gimbal_current_angles.copy()

    # Command a constant velocity
    action = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # max pitch velocity

    prev_angles = initial_angles.copy()
    for i in range(10):
        obs, reward, done, info = env.step(action)
        current_angles = env.gimbal_current_angles

        # In velocity mode, angles should change gradually
        delta = current_angles - prev_angles

        # Pitch should increase (positive velocity command)
        assert delta[0] > 0, f"Step {i}: Pitch should increase with positive velocity"

        prev_angles = current_angles.copy()

    # Final angle should be different from initial
    final_delta = env.gimbal_current_angles - initial_angles
    assert np.abs(final_delta[0]) > 0.01, "Gimbal should have moved significantly"

    env.close()
    print("✓ Velocity mode test PASSED")
    return True

def test_acceleration_mode():
    """Test acceleration mode"""
    print("\n" + "="*60)
    print("Testing ACCELERATION mode")
    print("="*60)

    env = gym.make('gimbal-curriculum-landing-aviary-v0',
                   gimbal_control_mode='acceleration',
                   gimbal_max_velocity=3.0,
                   gimbal_max_acceleration=10.0)
    obs = env.reset()

    # Initial state
    initial_angles = env.gimbal_current_angles.copy()
    initial_velocity = env.gimbal_current_velocity.copy()

    # Verify initial velocity is zero
    np.testing.assert_allclose(initial_velocity, np.zeros(3), rtol=1e-5)

    # Command a constant acceleration
    action = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # max pitch acceleration

    prev_velocity = initial_velocity.copy()
    for i in range(10):
        obs, reward, done, info = env.step(action)
        current_velocity = env.gimbal_current_velocity

        # In acceleration mode, velocity should increase
        delta_v = current_velocity - prev_velocity

        # Pitch velocity should increase (positive acceleration command)
        assert delta_v[0] > 0 or np.abs(current_velocity[0] - env.gimbal_max_velocity) < 1e-3, \
            f"Step {i}: Pitch velocity should increase or be at max"

        prev_velocity = current_velocity.copy()

    # Velocity should have increased from zero
    assert env.gimbal_current_velocity[0] > 0.01, "Velocity should have increased"

    # Angle should have changed
    final_delta = env.gimbal_current_angles - initial_angles
    assert final_delta[0] > 0.001, "Gimbal should have moved"

    env.close()
    print("✓ Acceleration mode test PASSED")
    return True

def test_angle_limits():
    """Test that gimbal angles are properly clamped in all modes"""
    print("\n" + "="*60)
    print("Testing ANGLE LIMITS across all modes")
    print("="*60)

    for mode in ['position', 'velocity', 'acceleration']:
        print(f"\n  Testing {mode} mode angle limits...")

        env = gym.make('gimbal-curriculum-landing-aviary-v0',
                       gimbal_control_mode=mode,
                       gimbal_max_velocity=10.0,  # High values to try to exceed limits
                       gimbal_max_acceleration=50.0)
        obs = env.reset()

        # Command maximum positive values
        action = np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

        # Run many steps to try to exceed limits
        for _ in range(100):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()

        # Check that angles are within [-1, 1]
        assert np.all(env.gimbal_current_angles >= -1.0 - 1e-6), \
            f"{mode}: Angles below -1.0: {env.gimbal_current_angles}"
        assert np.all(env.gimbal_current_angles <= 1.0 + 1e-6), \
            f"{mode}: Angles above 1.0: {env.gimbal_current_angles}"

        env.close()
        print(f"  ✓ {mode} mode angle limits OK")

    print("✓ Angle limits test PASSED for all modes")
    return True

def test_reset_consistency():
    """Test that reset works correctly in all modes"""
    print("\n" + "="*60)
    print("Testing RESET consistency across all modes")
    print("="*60)

    for mode in ['position', 'velocity', 'acceleration']:
        print(f"\n  Testing {mode} mode reset...")

        env = gym.make('gimbal-curriculum-landing-aviary-v0',
                       gimbal_control_mode=mode,
                       gimbal_max_velocity=3.0,
                       gimbal_max_acceleration=10.0)

        # First reset
        obs = env.reset()
        initial_angles_1 = env.gimbal_current_angles.copy()

        # Take some actions to change state
        for _ in range(20):
            action = np.random.uniform(-1, 1, 5).astype(np.float32)
            obs, reward, done, info = env.step(action)
            if done:
                break

        # Second reset
        obs = env.reset()
        initial_angles_2 = env.gimbal_current_angles.copy()

        # Both resets should give same initial angles
        np.testing.assert_allclose(
            initial_angles_1,
            initial_angles_2,
            rtol=1e-5,
            err_msg=f"{mode}: Reset should give consistent initial state"
        )

        # Velocity should be zero after reset (for acceleration mode)
        if mode == 'acceleration':
            np.testing.assert_allclose(
                env.gimbal_current_velocity,
                np.zeros(3),
                rtol=1e-5,
                err_msg=f"{mode}: Velocity should be zero after reset"
            )

        env.close()
        print(f"  ✓ {mode} mode reset OK")

    print("✓ Reset consistency test PASSED for all modes")
    return True

def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# GIMBAL CONTROL MODES TEST SUITE")
    print("#"*60)

    tests = [
        ("Position Mode (Backward Compatibility)", test_position_mode),
        ("Velocity Mode", test_velocity_mode),
        ("Acceleration Mode", test_acceleration_mode),
        ("Angle Limits", test_angle_limits),
        ("Reset Consistency", test_reset_consistency),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n✗ {name} FAILED with error: {e}")

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests PASSED! Gimbal control implementation is working correctly.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) FAILED. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
