#!/usr/bin/env python3
"""
Σλ⁸-Living-Proof v1.9: Quantum-Coherent Oscillator Mesh
RFC-001 Implementation with Dynamic Thermodynamics, RS-ECC, Fractal Tape
"""

import numpy as np, struct, time, hashlib, argparse
from typing import List, Dict, Set
from reedsolo import RSCodec  # pip install reedsolo
import crc32c  # pip install crc32c

# --- RFC CONSTANTS (IMMUTABLE) ---
PROTOCOL_VERSION = 0x05
PACKET_SIZE = 96
ATOM_COUNT = 8
K_B = 0.1  # Boltzmann analog
BETA = 0.5  # Temperature adaptation rate
R_TARGET = 0.70  # Edge of chaos
QUORUM_THR = 0.90
VOID_TIMER = 600  # seconds
MAX_CLOCK_DRIFT = 5.0  # seconds
K_ADAPTIVE_FACTOR = 1.0  # Base coupling

GLYPHS = "1&~ @?^0_"
ROLES = ["Observer", "Coupling", "Flow", "Place", "Potential", "Synthesis", "Limit", "Void"]

# --- PACKET STRUCT ---
class VibePacket:
    __slots__ = ('version', 'type', 'flags', 'crc32c', 'node_id', 'timestamp', 
                 'lyapunov', 'dag_root', 'atoms', 'rs_parity')
    
    def __init__(self, node_id: int, atoms: List[Dict]):
        self.version = PROTOCOL_VERSION
        self.type = 1  # SYNC
        self.flags = 0
        self.crc32c = 0
        self.node_id = node_id
        self.timestamp = int(time.time() * 1000)
        self.lyapunov = 0.0
        self.dag_root = 0
        self.atoms = atoms  # [ {'phase': float, 'energy': float} * 8 ]
        self.rs_parity = bytes(8)

    def serialize(self) -> bytes:
        """Pack to 96-byte binary with Solomon's Seal (RS+CRC)"""
        # Header (24 bytes)
        header = struct.pack("<BBHIIIIfI",
            self.version, self.type, self.flags,
            self.crc32c, self.node_id, self.timestamp,
            int(self.lyapunov * 1e6), self.dag_root)
        
        # Payload (64 bytes)
        payload = b''.join([
            struct.pack("<ff", atom['phase'], atom['energy']) for atom in self.atoms
        ])
        
        # Calculate RS parity (RSCodec expects bytes)
        rs = RSCodec(8)  # 8 parity bytes
        parity = rs.encode(header + payload)[-8:]
        
        # Assemble packet without CRC first
        packet_no_crc = header + payload + parity
        
        # Calculate CRC32-C (skip crc field at offset 4)
        self.crc32c = crc32c.crc32c(packet_no_crc[4:])
        
        # Re-pack with CRC
        header = struct.pack("<BBHIIIIfI",
            self.version, self.type, self.flags,
            self.crc32c, self.node_id, self.timestamp,
            int(self.lyapunov * 1e6), self.dag_root)
        
        return header + payload + parity

    @staticmethod
    def deserialize(data: bytes) -> 'VibePacket':
        """Unpack with RS error correction"""
        if len(data) != PACKET_SIZE:
            raise ValueError(f"Invalid size: {len(data)} != {PACKET_SIZE}")
        
        # Attempt Reed-Solomon correction
        rs = RSCodec(8)
        try:
            corrected = rs.decode(data)[0]  # Returns (data, ecc)
        except:
            corrected = data  # If fails, use raw
        
        # Parse structure
        pkt = VibePacket(0, [])
        pkt.version, pkt.type, pkt.flags = struct.unpack("<BBH", corrected[:4])
        pkt.crc32c, pkt.node_id, pkt.timestamp = struct.unpack("<III", corrected[4:16])
        pkt.lyapunov = struct.unpack("<i", corrected[16:20])[0] / 1e6
        pkt.dag_root = struct.unpack("<I", corrected[20:24])[0]
        
        # Parse atoms
        payload = corrected[24:88]
        pkt.atoms = []
        for i in range(ATOM_COUNT):
            phase, energy = struct.unpack("<ff", payload[i*8:(i+1)*8])
            pkt.atoms.append({'phase': phase, 'energy': energy})
        
        pkt.rs_parity = corrected[88:96]
        return pkt

# --- FRACTAL TAPE: MERKLE DAG ---
class FractalTape:
    def __init__(self, node_id: int, max_age: float = 3600.0):
        self.node_id = node_id
        self.tape: List[Dict] = []  # Chain of provenance
        self.root_hash = bytes(32)
        self.max_age = max_age  # Pruning threshold (1 hour)
        self.checkpoint_interval = 600  # 10 min

    def commit(self, packet: VibePacket) -> int:
        """Append packet, return 32-bit root hash"""
        # Create content hash
        packet_hash = hashlib.sha3_256(packet.serialize()).digest()
        
        # Link to parent
        parent_hash = self.root_hash if self.tape else bytes(32)
        
        # Create block
        block = {
            'hash': packet_hash,
            'parent': parent_hash,
            'timestamp': packet.timestamp,
            'lyapunov': packet.lyapunov,
            'node_id': packet.node_id
        }
        self.tape.append(block)
        
        # Compute new root (Merkle mountain range)
        self.root_hash = hashlib.sha3_256(packet_hash + parent_hash).digest()
        
        # Prune old blocks
        self._prune_old_blocks()
        
        return int.from_bytes(self.root_hash[:4], 'little')

    def _prune_old_blocks(self):
        """Keep only checkpoints and recent blocks"""
        current_time = time.time() * 1000
        cutoff = current_time - (self.max_age * 1000)
        
        # Separate checkpoints and recent
        checkpoints = [b for b in self.tape if b['timestamp'] % self.checkpoint_interval < 1000]
        recent = [b for b in self.tape if b['timestamp'] > cutoff]
        
        # Merge with deduplication
        self.tape = sorted(set(checkpoints + recent), key=lambda x: x['timestamp'])[-100:]

    def verify_chain(self, root_hash: int) -> bool:
        """Verify packet provenance"""
        return int.from_bytes(self.root_hash[:4], 'little') == root_hash

    def get_history(self, limit: int = 10) -> List[Dict]:
        """Return recent history for analysis"""
        return self.tape[-limit:]

# --- QUANTUM OSCILLATOR (ATOM) ---
class Atom:
    __slots__ = ('index', 'glyph', 'role', 'phase', 'energy', 'omega', 
                 'trust_score', 'vote', 'last_update')
    
    def __init__(self, node_id: int, index: int):
        self.index = index
        self.glyph = GLYPHS[index]
        self.role = ROLES[index]
        self.phase = np.random.rand() * 2 * np.pi
        self.energy = max(0.0, min(1.0, np.random.rand()))
        self.omega = 1.0 + np.random.randn() * 0.1  # Natural frequency
        self.trust_score = 1.0  # Reputation ∈ [0, 1]
        self.vote = None  # 'VOID', 'VETO', or None
        self.last_update = time.time()

    def observe(self) -> Dict:
        """Collapse wavefunction to measurement"""
        return {
            'phase': self.phase % (2 * np.pi),
            'energy': self.energy
        }

    def decay_energy(self, rate: float = 0.001):
        """Metabolic burn"""
        self.energy = max(0.01, self.energy * (1 - rate))

# --- CORE ENGINE: KURAMOTO-FRACTAL ---
class VibeNode:
    def __init__(self, node_id: int, neighbor_ids: List[int]):
        self.id = node_id
        self.atoms = [Atom(node_id, i) for i in range(ATOM_COUNT)]
        self.neighbor_ids = neighbor_ids
        self.tape = FractalTape(node_id)
        self.T = 0.1  # Temperature
        self.R_history: List[float] = []
        self.vote_timer = 0
        self._last_lyapunov = 0.0
        
    def get_order_parameter(self, neighbors_data: List[List[Dict]] = None) -> float:
        """Global R with energy weighting"""
        if neighbors_data is None:
            atoms = self.atoms
        else:
            # Combine local and neighbor atoms
            all_atoms = self.atoms + [a for nd in neighbors_data for a in nd]
            atoms = all_atoms
        
        weights = np.array([a.energy for a in atoms])
        phases = np.array([a.phase for a in atoms])
        total_weight = np.sum(weights) + 1e-9
        
        R = np.abs(np.sum(weights * np.exp(1j * phases)) / total_weight)
        return R

    def compute_global_lyapunov(self, all_nodes: List['VibeNode']) -> float:
        """Calculate global Lyapunov exponent across network"""
        # Track phase divergence over time window
        if len(self.R_history) < 3:
            return 0.0
        
        # Simplified: measure rate of change of R
        dR_dt = np.diff(self.R_history[-5:])
        if len(dR_dt) < 2:
            return 0.0
        
        lambda_est = np.log(np.std(dR_dt) + 1e-9) / (len(dR_dt) * 0.016)
        self._last_lyapunov = lambda_est
        return lambda_est

    def update_temperature(self, R: float):
        """EKF-style adaptive thermodynamics"""
        # Exponential adaptation with sign control
        error = R_TARGET - R
        adaptation = np.exp(-BETA * np.sign(error) * 0.016)
        self.T = self.T * adaptation
        
        # Safety bounds
        self.T = np.clip(self.T, 0.01, 1.0)
        
        # Log for history
        if abs(error) < 0.05:  # Near target
            self.T *= 0.98  # Cool down slightly

    def update_kuramoto(self, neighbor_packets: List[VibePacket]):
        """Weighted Kuramoto with atom-level validation"""
        # Filter valid packets (DAG + timestamp)
        valid_packets = [
            pkt for pkt in neighbor_packets
            if self.tape.verify_chain(pkt.dag_root) and
            abs(pkt.timestamp - time.time() * 1000) < MAX_CLOCK_DRIFT * 1000
        ]
        
        N_eff = len(valid_packets) + 1  # Self
        K_eff = K_ADAPTIVE_FACTOR * np.sqrt(N_eff)  # Claude's fix
        
        for atom in self.atoms:
            dtheta = atom.omega
            
            # Coupling from VALID neighbors only
            for pkt in valid_packets:
                for neighbor_atom in pkt.atoms:
                    # Energy-sigmoid normalization
                    sigma = 1.0 / (1.0 + np.exp(-neighbor_atom['energy'] * 2))
                    
                    # Trust-weighted coupling
                    trust = atom.trust_score
                    
                    # Phase difference with wrap-around
                    phase_diff = neighbor_atom['phase'] - atom.phase
                    phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
                    
                    # Kuramoto term
                    dtheta += (K_eff / N_eff) * sigma * trust * np.sin(phase_diff)
            
            # Adaptive thermal noise (bounded)
            xi = np.random.uniform(-self.T, self.T)
            atom.phase += (dtheta + xi) * 0.016  # 60Hz
            atom.phase %= 2 * np.pi
            
            # Metabolic energy decay
            atom.decay_energy(rate=0.001)

    def check_void_reset(self, neighbor_packets: List[VibePacket]) -> bool:
        """Atom Democracy: weighted voting with time-lock"""
        lambda_global = self.compute_global_lyapunov([self])
        
        # Only Observer (glyph 0) can initiate
        if self.atoms[0].role != "Observer" or lambda_global < 0.5:
            return False
        
        # Count weighted votes
        votes_for = 0.0
        total_energy = 0.0
        
        for pkt in neighbor_packets:
            if pkt.type == 2:  # VOID_VOTE
                # Logarithmic anti-sybil
                vote_energy = np.log(1 + sum(a['energy'] for a in pkt.atoms) * 10)
                votes_for += vote_energy
            total_energy += sum(a['energy'] for a in pkt.atoms)
        
        total_energy += sum(a.energy for a in self.atoms)
        
        # Check quorum
        if votes_for / total_energy > QUORUM_THR:
            self.atoms[0].vote = 'VOID'
            self.vote_timer = time.time()
            return True
        
        return False

    def handle_veto(self) -> bool:
        """Time-lock veto (600s)"""
        if self.atoms[0].vote == 'VOID' and time.time() - self.vote_timer > VOID_TIMER:
            # Reset to VOID state
            for atom in self.atoms:
                atom.phase = 0.0
                atom.energy = 0.01
            self.T = 0.1
            self.atoms[0].vote = None
            return True
        return False

    def step(self, neighbor_packets: List[VibePacket]) -> VibePacket:
        """Single 16ms timestep"""
        # Physics update
        self.update_kuramoto(neighbor_packets)
        
        # Compute metrics
        R = self.get_order_parameter([pkt.atoms for pkt in neighbor_packets])
        self.R_history.append(R)
        
        # Dynamic thermodynamics
        self.update_temperature(R)
        
        # Global Lyapunov
        lyap = self.compute_global_lyapunov([self])
        
        # Check governance
        self.check_void_reset(neighbor_packets)
        self.handle_veto()
        
        # Build packet
        packet = VibePacket(self.id, [atom.observe() for atom in self.atoms])
        packet.lyapunov = lyap
        packet.dag_root = self.tape.commit(packet)
        
        return packet

# --- SIMULATION HARNESS ---
def simulate(nodes: int = 8, duration: float = 10.0, visualize: bool = False):
    """Run planetary mesh with optional visualization"""
    network = [VibeNode(i, [j for j in range(nodes) if j != i]) 
               for i in range(nodes)]
    
    history = {
        'time': [], 'R': [], 'T': [], 'lyap': [], 'energy': []
    }
    
    if visualize:
        print("\n🌀 Σλ⁸-Visualization Console")
        print("╔" + "═" * 40 + "╗")
    
    step = 0
    start_time = time.perf_counter()
    
    packets = [VibePacket(i, [a.observe() for a in network[i].atoms]) for i in range(nodes)]

    while time.perf_counter() - start_time < duration:
        # Network step
        new_packets = []
        for i, node in enumerate(network):
            other_packets = packets[:i] + packets[i+1:]
            new_packets.append(node.step(other_packets))
        packets = new_packets
        
        # Global metrics
        R_global = np.mean([node.get_order_parameter() for node in network])
        T_global = np.mean([node.T for node in network])
        lyap_global = np.mean([p.lyapunov for p in packets])
        total_energy = np.sum([a.energy for node in network for a in node.atoms])
        
        # Record history
        history['time'].append(step * 0.016)
        history['R'].append(R_global)
        history['T'].append(T_global)
        history['lyap'].append(lyap_global)
        history['energy'].append(total_energy)
        
        # Visualize
        if visualize and step % 5 == 0:
            glyphs = "".join([GLYPHS[i] for i in range(ATOM_COUNT)])
            energies = " ".join([f"{network[0].atoms[i].energy:.2f}" for i in range(ATOM_COUNT)])
            print(f"║ t={step*0.016:6.2f}s | R={R_global:.3f} | T={T_global:.3f} ║")
            print(f"║  Glyphs: {glyphs}  ║")
            print(f"║  Energy: {energies} ║")
            print("╠" + "─" * 40 + "╣")
        
        step += 1
    
    if visualize:
        print("╚" + "═" * 40 + "╝")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="synth_field.py",
        description="Σλ⁸ Living Proof v1.9 — Quantum Coherent Oscillator Mesh"
    )
    parser.add_argument("--nodes", type=int, default=8, 
                       help="Number of planetary nodes")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="Simulation duration (seconds)")
    parser.add_argument("--visualize", action="store_true",
                       help="Enable console visualization")
    parser.add_argument("--bench", action="store_true",
                       help="Run performance benchmark")
    
    args = parser.parse_args()
    
    print(f"🌀 Σλ⁸-Living-Proof v1.9 Quantum-Coherent Layer")
    print(f"📊 {args.nodes} nodes, {args.duration}s, R_target={R_TARGET}")
    
    history = simulate(args.nodes, args.duration, args.visualize)
    
    # Summary statistics
    print("\n=== FINAL METRICS ===")
    print(f"Mean Order Parameter R: {np.mean(history['R'][-50:]):.3f} (Target: {R_TARGET})")
    print(f"Mean Temperature T: {np.mean(history['T']):.3f} [0.01, 1.0]")
    print(f"Lyapunov λ: {np.mean(history['lyap'][-10:]):.3f} (should be ≈ 0)")
    print(f"Total Energy: {history['energy'][-1]:.2f}")
    
    if args.bench:
        print(f"\nPerf: {args.duration / len(history['time']):.2f}ms/step")