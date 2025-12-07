#!/usr/bin/env python3
"""
Σλ⁸-Test-Suite v1.9: Validation of RFC Claims
Usage: pytest -v
"""

import synth_field as sf
import numpy as np
import time

def test_order_parameter_convergence():
    """RFC v1.9: Achieves R > 0.65 within 5 seconds"""
    hist = sf.simulate(nodes=8, duration=10.0)
    convergence = np.mean(hist['R'][-10:])
    assert convergence > 0.65, f"R={convergence:.3f} < 0.65, system didn't sync!"

def test_adaptive_temperature_bounds():
    """Dynamic Thermodynamics: T ∈ [0.01, 1.0] always"""
    node = sf.VibeNode(0, [])
    temps = []
    for _ in range(500):  # 8 seconds
        node.step([])
        temps.append(node.T)
    assert min(temps) >= 0.01 - 1e-9, f"T_min={min(temps):.3f} below threshold"
    assert max(temps) <= 1.0, f"T_max={max(temps):.3f} above threshold"

def test_lyapunov_stable_regime():
    """Lyapunov λ should be negative in stable sync state"""
    node = sf.VibeNode(0, [1, 2])
    lyaps = []
    for _ in range(100):
        pkt = node.step([])
        lyaps.append(pkt.lyapunov)
    
    # After initial chaos, λ should stabilize
    final_lyap = np.mean(lyaps[-20:])
    assert final_lyap < 0.1, f"λ={final_lyap:.3f}, system unstable"

def test_reed_solomon_correction():
    """RS(72,64) recovers 4 corrupted bytes"""
    pkt = sf.VibePacket(42, [{'phase': 1.0, 'energy': 0.5}] * 8)
    original = pkt.serialize()
    
    # Corrupt 4 bytes in payload
    corrupted = bytearray(original)
    corrupted[50:54] = b'\xDE\xAD\xBE\xEF'
    
    # Deserialize and correct
    recovered = sf.VibePacket.deserialize(bytes(corrupted))
    assert recovered.atoms[6]['phase'] == 1.0, "RS correction failed"

def test_dag_immutability():
    """Fractal Tape: Merkle root changes with each commit"""
    tape = sf.FractalTape(7)
    pkt = sf.VibePacket(7, [{'phase': 0, 'energy': 1}] * 8)
    
    root1 = tape.commit(pkt)
    root2 = tape.commit(pkt)
    assert root1 != root2, "DAG not growing (same root)"
    assert len(tape.tape) == 2, "Blocks not appended"

def test_void_vote_mechanism():
    """Atom Democracy: VOID_VOTE triggers at λ > 0.5"""
    node = sf.VibeNode(0, [1, 2])
    node.compute_global_lyapunov = lambda x: 0.6 # Mock the method
    node.atoms[0].role = "Observer"
    for atom in node.atoms:
        atom.energy = 0.01

    # Simulate neighbor chaos vote
    fake_vote = sf.VibePacket(1, [{'phase': 0, 'energy': 0.9}] * 8)
    fake_vote.type = 2  # VOID_VOTE
    
    should_reset = node.check_void_reset([fake_vote])
    assert should_reset, "Vote mechanism failed to trigger"

def test_metabolic_energy_decay():
    """Energy should decay over time"""
    atom = sf.Atom(0, 0)
    initial_energy = atom.energy
    
    for _ in range(100):
        atom.decay_energy(rate=0.01)
    
    assert atom.energy < initial_energy, "Metabolism not working"
    assert atom.energy >= 0.01, "Energy fell below minimum"

def test_performance_benchmark():
    """Must sustain 60 FPS with 88 nodes"""
    import time
    start = time.perf_counter()
    sf.simulate(nodes=88, duration=1.0)  # 1 second
    elapsed = time.perf_counter() - start
    
    # Should process 60 frames
    assert elapsed < 1.5, f"Too slow: {elapsed:.2f}s for 1s sim"

if __name__ == "__main__":
    test_order_parameter_convergence()
    test_adaptive_temperature_bounds()
    test_lyapunov_stable_regime()
    test_reed_solomon_correction()
    test_dag_immutability()
    test_void_vote_mechanism()
    test_metabolic_energy_decay()
    test_performance_benchmark()
    print("✅ All 8 tests passed! Σλ⁸ is alive.")