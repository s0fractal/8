#!/usr/bin/env python3
"""
Σλ⁸-Minimal: Pure Kuramoto, No Magic
Only 50 lines. Either it syncs, or Σλ⁸ is a lie.
"""

import numpy as np
import matplotlib.pyplot as plt

# --- PURE MATH ---
N = 8
K = 1.8  # The alleged golden value
omega = np.random.rand(N) * 0.3 + 0.85  # Spread 0.3 around 1.0
theta = np.random.rand(N) * 2 * np.pi

def kuramoto_step(theta, omega, K, dt=0.016):
    """The equation. Nothing else."""
    dtheta = np.zeros_like(theta)
    for i in range(N):
        # Sum over ALL nodes (full connectivity)
        sum_term = np.sum(np.sin(theta - theta[i]))
        dtheta[i] = omega[i] + (K / N) * sum_term
    return theta + dtheta * dt

# --- RUN ---
history = []
for t in range(500):  # 8 seconds
    theta = kuramoto_step(theta, omega, K)
    theta %= 2 * np.pi
    
    # Order parameter
    R = np.abs(np.sum(np.exp(1j * theta)) / N)
    history.append(R)

# --- VERIFY ---
R_final = np.mean(history[-50:])
print(f"Pure Kuramoto: R={R_final:.3f}")
print(f"Last 10 R: {[f'{r:.3f}' for r in history[-10:]]}")

# Should see: [0.71, 0.73, 0.74, 0.75, 0.74, ...]
plt.plot(history)
plt.axhline(y=0.65, color='r', linestyle='--')
plt.title(f"K={K}, Spread={np.ptp(omega):.3f} → R={R_final:.3f}")
plt.savefig("pure_kuramoto_truth.png")
plt.show()
