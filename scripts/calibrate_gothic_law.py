"""Demonstrate the reduced Gothic-arch calibration pipeline end to end.

A few lines turn a physical groove into a verified, ready-to-use reduced contact
law, then print its describe() report (coefficients, accuracy, speed-up). Run it:

    uv run python scripts/calibrate_gothic_law.py
"""

from __future__ import annotations

import hertzian


def main() -> None:
    """Calibrate the README bearing groove and print the verification report."""
    spec = hertzian.GrooveSpec(
        ball_radius=4.0e-3,
        tube_radius=4.16e-3,
        centre_radius=15.0e-3,
        centre_offset=65.0e-6,
        e_star=100.0e9,
    )
    cal = hertzian.calibrate(spec)
    print(cal.describe())

    # The calibrated law is ready for a multibody inner loop.
    f_t, f_n = cal.law.force(1.0e-6, 6.0e-6)
    print(f"\nforce(delta_t=1e-6, delta_n=6e-6) = ({f_t:.3f}, {f_n:.3f}) N")


if __name__ == "__main__":
    main()
