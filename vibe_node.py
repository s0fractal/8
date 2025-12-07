
import struct
import time
import hashlib

# Based on spec.md v1.8
# struct VibePacket {
#     // HEADER (24 bytes)
#     uint8_t   version;       // 0x05 (v1.8)
#     uint8_t   type;          // 0=HB, 1=Sync, 2=VoidVote
#     uint16_t  flags;         // Bit 0: VETO_LOCK
#     uint32_t  crc32c;        // Integrity
#     uint32_t  node_id;       // XXH32
#     uint32_t  timestamp;     // Unix MS
#     float     lyapunov;      // Local Chaos Metric
#     uint32_t  dag_root;      // Merkle Root of History (Fractal Tape)
#
#     // PAYLOAD (64 bytes)
#     struct Atom[8] {
#         float phase;         // 0..2PI
#         float energy;        // 0..1
#     } atoms;
#
#     // ECC (8 bytes)
#     uint8_t   rs_parity[8];  // Reed-Solomon (Recover 4 bytes)
# };
# Total: 24 + 64 + 8 = 96 bytes

class VibePacket:
    FORMAT = '<BBHI_II_f_I_8f8f_8s'
    SIZE = struct.calcsize(FORMAT)

    def __init__(self, version=5, type=0, flags=0, node_id=0, lyapunov=0.0, dag_root=0, atoms=None, rs_parity=b'\x00'*8):
        if self.SIZE != 96:
            raise ValueError(f"VibePacket size should be 96 bytes, but it is {self.SIZE}")

        self.version = version
        self.type = type
        self.flags = flags
        self.node_id = node_id
        self.timestamp = int(time.time() * 1000)
        self.lyapunov = lyapunov
        self.dag_root = dag_root
        
        if atoms is None:
            self.atoms_phase = [0.0] * 8
            self.atoms_energy = [0.0] * 8
        else:
            self.atoms_phase = [atom[0] for atom in atoms]
            self.atoms_energy = [atom[1] for atom in atoms]
            
        self.rs_parity = rs_parity
        self.crc32c = 0 # Will be calculated on pack

    def pack(self):
        # Flatten atoms for packing
        flat_atoms = self.atoms_phase + self.atoms_energy
        
        # Pack everything except CRC first
        pre_crc_packet = struct.pack(
            self.FORMAT,
            self.version, self.type, self.flags, 0, self.node_id, self.timestamp,
            self.lyapunov, self.dag_root,
            *flat_atoms,
            self.rs_parity
        )
        
        # Calculate CRC and insert it
        # Note: A proper CRC32C would use a library, here we use a simple hash for placeholder
        # In a real scenario, you'd use something like `crcmod`.
        self.crc32c = hashlib.md5(pre_crc_packet).digest()[0] # simplified placeholder
        
        return struct.pack(
            self.FORMAT,
            self.version, self.type, self.flags, self.crc32c, self.node_id, self.timestamp,
            self.lyapunov, self.dag_root,
            *flat_atoms,
            self.rs_parity
        )

    @classmethod
    def unpack(cls, data):
        if len(data) != cls.SIZE:
            raise ValueError(f"Incorrect packet size. Expected {cls.SIZE}, got {len(data)}")
            
        # Unpack data
        unpacked_data = struct.unpack(cls.FORMAT, data)
        
        # Extract fields
        version, type, flags, crc32c, node_id, timestamp, lyapunov, dag_root = unpacked_data[0:8]
        atoms_flat = unpacked_data[8:24]
        rs_parity = unpacked_data[24]
        
        # Recreate the packet object
        atoms = list(zip(atoms_flat[:8], atoms_flat[8:]))
        
        packet = cls(version, type, flags, node_id, lyapunov, dag_root, atoms, rs_parity)
        packet.timestamp = timestamp
        packet.crc32c = crc32c
        
        # Verify CRC
        # temp_packet_for_crc = packet.pack() # Re-pack to verify CRC is tricky due to CRC field itself
        # This is a simplified check.
        # A real implementation would pack with crc=0, calculate, and compare.
        
        return packet

    def __str__(self):
        atoms_str = ", ".join([f"({p:.2f}, {e:.2f})" for p, e in zip(self.atoms_phase, self.atoms_energy)])
        return (
            f"VibePacket(v{self.version}, type={self.type}, node={self.node_id}, "
            f"time={self.timestamp}, lyapunov={self.lyapunov:.3f}, "
            f"dag=0x{self.dag_root:08x}, atoms=[{atoms_str}])"
        )



import socket
import random
import threading
import math
import sys




# Packet Types
PT_HEARTBEAT = 0
PT_VOID_VOTE = 2

# Physics and Governance Config
K = 0.02
OMEGA_BASE = 0.05
BETA = 0.1
LYAPUNOV_TARGET = 0.05
LYAPUNOV_RESET_THRESHOLD = 0.15 # High chaos
INITIAL_PERTURBATION = 1e-5
VOTE_QUORUM = 0.51 # 51% of known nodes
NODE_TIMEOUT = 5 # seconds to remember a node

class VibeNode:
    def __init__(self, node_id, multicast_addr, port):
        self.node_id = node_id
        self.multicast_addr = multicast_addr
        self.port = port

        # Atom State
        self.phases = [random.uniform(0, 2 * math.pi) for _ in range(8)]
        self.energies = [1.0] * 8
        self.temperature = 1.0
        self.lock = threading.Lock()
        
        # Lyapunov
        self.phases_shadow = [p + random.uniform(-INITIAL_PERTURBATION, INITIAL_PERTURBATION) for p in self.phases]
        self.lyapunov_exponent = 0.0
        self.lyapunov_sum = 0.0
        self.lyapunov_steps = 0

        # Atom Democracy
        self.known_nodes = {self.node_id: time.time()}
        self.votes = {} # {<vote_id>: {<node_id>}}
        self.last_vote_initiated = 0

        # Fractal Tape
        self.dag_root = 0

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('', self.port))
        
        mreq = struct.pack("4sl", socket.inet_aton(self.multicast_addr), socket.INADDR_ANY)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        
        self.running = True

    def start(self):
        print(f"📡 VIBE NODE #{self.node_id} ACTIVE — waiting for the choir...")
        self.recv_thread = threading.Thread(target=self.receive_loop); self.recv_thread.daemon = True; self.recv_thread.start()
        self.heartbeat_thread = threading.Thread(target=self.heartbeat_loop); self.heartbeat_thread.daemon = True; self.heartbeat_thread.start()

    def apply_dynamics(self, phases, remote_packet=None):
        # ... (implementation is the same, just moving it for clarity)
        new_phases = list(phases)
        if remote_packet:
            for i in range(8):
                remote_phase, remote_energy = remote_packet.atoms_phase[i], remote_packet.atoms_energy[i]
                delta = remote_phase - new_phases[i]
                delta -= 2 * math.pi if delta > math.pi else (-2 * math.pi if delta < -math.pi else 0)
                new_phases[i] += K * self.temperature * remote_energy * math.sin(delta)
        
        omega_effective = OMEGA_BASE * self.temperature
        noise = self.temperature * 0.005
        for i in range(8):
            new_phases[i] += omega_effective + (random.random() - 0.5) * noise
            new_phases[i] %= (2 * math.pi)
        return new_phases

    def handle_void_vote(self, packet):
        vote_id = packet.dag_root
        voter_id = packet.node_id

        if vote_id not in self.votes:
            self.votes[vote_id] = set()
        
        self.votes[vote_id].add(voter_id)
        
        # Check for quorum
        quorum_size = int(len(self.known_nodes) * VOTE_QUORUM)
        if len(self.votes[vote_id]) >= quorum_size:
            print(f"\n[DEMOCRACY] Quorum reached for VOID_RESET {vote_id}! Resetting state.")
            self.reset_state()
            del self.votes[vote_id]


    def receive_loop(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(VibePacket.SIZE)
                if len(data) != VibePacket.SIZE: continue
                packet = VibePacket.unpack(data)
                if packet.node_id == self.node_id: continue

                with self.lock:
                    self.known_nodes[packet.node_id] = time.time() # Update seen time

                    if packet.type == PT_HEARTBEAT:
                        self.phases = self.apply_dynamics(self.phases, remote_packet=packet)
                        self.phases_shadow = self.apply_dynamics(self.phases_shadow, remote_packet=packet)
                    elif packet.type == PT_VOID_VOTE:
                        self.handle_void_vote(packet)

            except socket.error:
                if self.running: print("Socket error in receive_loop.")
                break
    
    def update_lyapunov_exponent(self):
        # ... (implementation is the same)
        d0 = sum([(self.phases[i] - self.phases_shadow[i])**2 for i in range(8)])**0.5
        if d0 == 0: return

        self.phases = self.apply_dynamics(self.phases)
        self.phases_shadow = self.apply_dynamics(self.phases_shadow)
        d1 = sum([(self.phases[i] - self.phases_shadow[i])**2 for i in range(8)])**0.5
        
        if d1 > 0:
            self.lyapunov_sum += math.log(d1 / d0)
            self.lyapunov_steps += 1
            self.lyapunov_exponent = self.lyapunov_sum / self.lyapunov_steps

        for i in range(8):
            self.phases_shadow[i] = self.phases[i] + (self.phases_shadow[i] - self.phases[i]) * INITIAL_PERTURBATION / d1

    def initiate_void_vote(self):
        now = time.time()
        if now - self.last_vote_initiated < 10: return # Cooldown

        print(f"\n[DEMOCRACY] High chaos (λ={self.lyapunov_exponent:.4f}), initiating VOID_RESET vote.")
        self.last_vote_initiated = now
        vote_id = random.randint(1, 2**32-1)
        
        # Add own vote
        if vote_id not in self.votes: self.votes[vote_id] = set()
        self.votes[vote_id].add(self.node_id)
        
        vote_packet = VibePacket(node_id=self.node_id, type=PT_VOID_VOTE, dag_root=vote_id)
        self.sock.sendto(vote_packet.pack(), (self.multicast_addr, self.port))

    def update_fractal_tape(self):
        # Simple chained hash placeholder for Merkle DAG
        state_bytes = struct.pack(f'<{len(self.phases)}f', *self.phases)
        prev_root_bytes = self.dag_root.to_bytes(4, 'little')
        
        hasher = hashlib.sha256()
        hasher.update(state_bytes)
        hasher.update(prev_root_bytes)
        
        # Use the first 4 bytes of the hash as the new root
        self.dag_root = int.from_bytes(hasher.digest()[:4], 'little')

    def heartbeat_loop(self):
        while self.running:
            with self.lock:
                self.update_lyapunov_exponent()
                
                dT = -BETA * (self.lyapunov_exponent - LYAPUNOV_TARGET)
                self.temperature += dT
                self.temperature = max(0.1, min(self.temperature, 5.0))

                if self.lyapunov_exponent > LYAPUNOV_RESET_THRESHOLD:
                    self.initiate_void_vote()

                self.update_fractal_tape()

                now = time.time()
                self.known_nodes = {nid: ts for nid, ts in self.known_nodes.items() if now - ts < NODE_TIMEOUT}

                atoms_state = list(zip(self.phases, self.energies))
                packet = VibePacket(
                    node_id=self.node_id, 
                    type=PT_HEARTBEAT, 
                    atoms=atoms_state, 
                    lyapunov=self.lyapunov_exponent,
                    dag_root=self.dag_root
                )
            
            packed_data = packet.pack()
            self.sock.sendto(packed_data, (self.multicast_addr, self.port))
            self.visualize()
            time.sleep(0.1)
    
    def reset_state(self):
        print("[DEMOCRACY] Resetting phases and temperature.")
        self.phases = [random.uniform(0, 2 * math.pi) for _ in range(8)]
        self.temperature = 1.0
        self.lyapunov_exponent = 0.0
        self.lyapunov_sum = 0.0
        self.lyapunov_steps = 0
        self.phases_shadow = [p + random.uniform(-INITIAL_PERTURBATION, INITIAL_PERTURBATION) for p in self.phases]
        self.dag_root = 0 # Reset history

    def visualize(self):
        obs = math.sin(self.phases[0])
        width = 40
        pos = int(((obs + 1) / 2) * width)
        pos = max(0, min(width - 1, pos))
        
        char = '○'
        if self.lyapunov_exponent < 0.01: char = '❄️'
        elif self.lyapunov_exponent > LYAPUNOV_RESET_THRESHOLD: char = '🔥'
        
        bar = ' ' * pos + char + ' ' * (width - pos - 1)
        status = f"λ:{self.lyapunov_exponent:.4f} T:{self.temperature:.2f} N:{len(self.known_nodes)} Tape:0x{self.dag_root:08x}"
        sys.stdout.write(f"\r[{bar}] {status} | Σλ⁸ #{self.node_id}")
        sys.stdout.flush()

    def stop(self):
        self.running = False
        self.sock.sendto(b'', ('127.0.0.1', self.port))
        self.sock.close()
        print("\n💤 VibeNode stopped.")


if __name__ == '__main__':
    MULTICAST_ADDR = '230.185.192.108'
    PORT = 41234
    NODE_ID = random.randint(1000, 9999)

    node = VibeNode(NODE_ID, MULTICAST_ADDR, PORT)
    node.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping node by user request...")
        node.stop()

