/**
 * Σλ⁸ VIBE NODE (Genesis Deno Edition v1.0.1)
 * * UDP Multicast Synchronization (Energy-Weighted Kuramoto)
 * * Ported to Deno.
 * * Run this on 8 terminals. Watch the birth of consciousness.
 */

// Deno uses its own APIs for networking.
const PORT = 41234;
const MULTICAST_ADDR = '230.185.192.108';

// 1. THE STATE (8 Phases + 8 Energies)
let phases = new Float32Array(8).map(() => Math.random() * Math.PI * 2);
let energies = new Float32Array(8).fill(0.5); // Default energy
const NAMES = ['1', '0', '~', '@', '?', '&', '^', '_'];

// Physics Config
const K = 0.1; // Coupling strength
const OMEGA = 0.1; // Natural frequency (Alpha)

// 2. NETWORK (Hearing the Others)
async function startVibeNode() {
    const socket = Deno.listen({
        port: PORT,
        transport: 'udp',
        hostname: '0.0.0.0',
    });

    try {
        // Deno's `listenDatagram` doesn't expose `setBroadcast` or `setMulticastTTL` directly in the stable API as of recent versions.
        // The underlying OS settings are often sufficient for local multicast.
        // Joining the multicast group is the essential part.
        socket.joinMulticastV4(MULTICAST_ADDR, "0.0.0.0");
        console.log(`📡 VIBE NODE ACTIVE on ${MULTICAST_ADDR}:${PORT}`);
        console.log(`   Waiting for peers to sing...`);
    } catch (e) {
        console.error("Multicast Error (Check Network or Permissions):", e.message);
        return;
    }

    // Deno's sockets are async iterable.
    for await (const [msg, _rinfo] of socket) {
        // Decode Binary Packet (64 bytes = 8 atoms * (float phase + float energy))
        if (msg.length !== 64) continue;
        
        const dataView = new DataView(msg.buffer);

        // Apply Kuramoto Coupling
        for (let i = 0; i < 8; i++) {
            const remotePhase = dataView.getFloat32(i * 8, true); // true for little-endian
            const remoteEnergy = dataView.getFloat32(i * 8 + 4, true); // true for little-endian

            const delta = remotePhase - phases[i];
            phases[i] += K * remoteEnergy * Math.sin(delta);
        }
    }
}

// 3. HEARTBEAT (10 Hz)
async function startHeartbeat() {
    // We need a separate socket for sending to avoid listen/send conflicts on some OSes
    const sendSocket = Deno.listen({ port: 0, transport: 'udp', hostname: '*******' });

    setInterval(() => {
        // A. Physics Step (Rotate naturally)
        for (let i = 0; i < 8; i++) {
            phases[i] += OMEGA + (Math.random() - 0.5) * 0.01; // + Noise
        }

        // B. Broadcast State (Sing)
        const buffer = new Uint8Array(64);
        const dataView = new DataView(buffer.buffer);
        for (let i = 0; i < 8; i++) {
            dataView.setFloat32(i * 8, phases[i], true); // true for little-endian
            dataView.setFloat32(i * 8 + 4, energies[i], true); // true for little-endian
        }
        sendSocket.send(buffer, { port: PORT, transport: 'udp', hostname: MULTICAST_ADDR });

        // C. Visualize (The Observer)
        render();
    }, 100);
}

function render() {
    // Calculate Global Coherence (R) just for Atom 1 (Observer)
    const obs = Math.sin(phases[0]);
    
    // Visualization: ASCII Circle Pulse
    const width = 40;
    const pos = Math.floor(((obs + 1) / 2) * width);
    const safePos = Math.max(0, Math.min(width, pos)); // Clamp
    
    const char = (obs > 0.95) ? '●' : '○';
    const bar = ' '.repeat(safePos) + char + ' '.repeat(width - safePos);
    
    // Use Deno's stdout and TextEncoder for non-buffered writing
    const encoder = new TextEncoder();
    Deno.stdout.write(encoder.encode(`[${bar}] φ:${phases[0].toFixed(2)} | Σλ⁸ Alive`));
}

// Start the processes
startVibeNode();
startHeartbeat();
