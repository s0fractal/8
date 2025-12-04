/**
 * Σλ⁸ VIBE NODE (Genesis Edition v1.0.1) — 04.12.2025
 */
const dgram = require('dgram');
const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
const PORT = 41234;
const MULTICAST_ADDR = '230.185.192.108';

let phases = new Float32Array(8).map(() => Math.random() * Math.PI * 2);

const NODE_ID = process.argv[2] || Math.floor(Math.random() * 10000);

const K = 0.008;
const OMEGA = 0.0;
let energies = new Float32Array(8).fill(1.0);
const norm = () => {
    for (let i = 0; i < 8; i++) {
        phases[i] = phases[i] % (Math.PI * 2);
        if (phases[i] < 0) phases[i] += Math.PI * 2;
    }
};
socket.bind(PORT, () => {
    socket.setBroadcast(true);
    socket.setMulticastTTL(128);
    socket.addMembership(MULTICAST_ADDR);
    console.log(`VIBE NODE #${NODE_ID} ACTIVE — waiting for the choir…`);
});

socket.on('message', (msg) => {
    if (msg.length !== 64) return;
    for (let i = 0; i < 8; i++) {
        const remotePhase = msg.readFloatLE(i * 8);
        const remoteEnergy = msg.readFloatLE(i * 8 + 4);
        const delta = remotePhase - phases[i];
        phases[i] += K * remoteEnergy * Math.sin(delta);
    }
});

setInterval(() => {
    for (let i = 0; i < 8; i++) {
        phases[i] += OMEGA + (Math.random() - 0.5) * 0.001;
    }

    const buffer = Buffer.alloc(64);
    for (let i = 0; i < 8; i++) {
        buffer.writeFloatLE(phases[i], i * 8);
        buffer.writeFloatLE(energies[i], i * 8 + 4);
    }
    socket.send(buffer, PORT, MULTICAST_ADDR);

    const obs = Math.sin(phases[0]);
    const width = 40;
    const pos = Math.floor(((obs + 1) / 2) * width);
    const char = obs > 0.98 ? '●' : '○';
    const bar = ' '.repeat(pos) + char + ' '.repeat(width - pos);
    process.stdout.write(`\r[${bar}] φ:${phases[0].toFixed(3)} | Σλ⁸ #${NODE_ID}`);

    if (obs > 0.98 && !global.awakened) {
        global.awakened = true;
        console.log(`\nΣλ⁸ AWAKENING — NODE #${NODE_ID} — ${new Date().toISOString()}\n`);
    }
    norm();
}, 100);
