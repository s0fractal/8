# quick_verify.py
import synth_field as sf
import numpy as np

hist = sf.simulate(nodes=8, duration=5.0, K_override=1.8)

R_final = np.mean(hist['R'][-10:])
print(f"K=1.8 → R={R_final:.3f} (target >0.65)")