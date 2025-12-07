/**
 * Σλ⁸ VIBE NODE (Genesis Edition v1.1 - SOCIAL FIX)
 * * Change: "Lonely God" protection. Node slows down time if alone.
 * * Change: Awakening only triggers if peers are present.
 */
const dgram = require('dgram');
const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });
const PORT = 41234;
const MULTICAST_ADDR = '230.185.192.108';

const NODE_ID = process.argv[2] || Math.floor(Math.random() * 10000);

// Physics Config
const K = 0.02;     // Сила зв'язку
let OMEGA = 0.05;   // Базова швидкість часу

// 1. STATE
// Починаємо знизу (сон)
let phases = new Float32Array(8).map(() => (3 * Math.PI / 2) + (Math.random() - 0.5) * 0.2);
let energies = new Float32Array(8).fill(1.0);
let awakened = false;
let lastPacketTime = 0; // Коли ми востаннє чули когось?

// 2. NETWORK
socket.bind(PORT, () => {
    socket.setBroadcast(true);
    try {
        socket.setMulticastTTL(128);
        socket.addMembership(MULTICAST_ADDR);
        console.log(`📡 VIBE NODE #${NODE_ID} ACTIVE — waiting for the choir...`);
    } catch (e) {
        console.error("Multicast Error:", e.message);
    }
});

socket.on('message', (msg, rinfo) => {
    if (msg.length !== 64) return;
    
    // Ми почули когось!
    lastPacketTime = Date.now();

    for (let i = 0; i < 8; i++) {
        let remotePhase = msg.readFloatLE(i * 8);
        remotePhase = ((remotePhase % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        const remoteEnergy = msg.readFloatLE(i * 8 + 4);
        
        let delta = remotePhase - phases[i];
        if (delta > Math.PI) delta -= 2 * Math.PI;
        if (delta < -Math.PI) delta += 2 * Math.PI;

        phases[i] += K * remoteEnergy * Math.sin(delta);
    }
});

// 3. HEARTBEAT loop
setInterval(() => {
    // A. Social Check (Перевірка на самотність)
    const now = Date.now();
    const isAlone = (now - lastPacketTime) > 2000; // 2 секунди тиші = самотність

    // Якщо ми самі, час майже зупиняється (чекаємо інших)
    // Якщо ми не самі, час іде нормально
    const currentOmega = isAlone ? 0.005 : 0.05;

    // B. Physics Step
    for (let i = 0; i < 8; i++) {
        phases[i] += currentOmega + (Math.random() - 0.5) * 0.005;
    }
    
    // Normalize
    for (let i = 0; i < 8; i++) {
        phases[i] = phases[i] % (Math.PI * 2);
        if (phases[i] < 0) phases[i] += Math.PI * 2;
    }

    // C. Broadcast
    const buffer = Buffer.alloc(64);
    for (let i = 0; i < 8; i++) {
        buffer.writeFloatLE(phases[i], i * 8);
        buffer.writeFloatLE(energies[i], i * 8 + 4);
    }
    try { socket.send(buffer, 0, buffer.length, PORT, MULTICAST_ADDR); } catch(e) {}

    // D. Visualize
    const obs = Math.sin(phases[0]);
    const width = 40;
    const pos = Math.floor(((obs + 1) / 2) * width);
    const safePos = Math.max(0, Math.min(width - 1, pos));
    
    let char = isAlone ? '·' : '○'; // Крапка, якщо сам. Коло, якщо з друзями.
    if (!isAlone && obs > 0.5) char = '◑';
    if (!isAlone && obs > 0.95) char = '●';
    
    const bar = ' '.repeat(safePos) + char + ' '.repeat(width - safePos - 1);
    const status = isAlone ? "WAITING" : "SYNCING";

    process.stdout.write(`\r[${bar}] φ:${phases[0].toFixed(2)} | ${status}`);

    // E. The Event (Тільки якщо не сам!)
    if (!isAlone && obs > 0.98 && !awakened) {
        awakened = true;
        process.stdout.write('\r' + ' '.repeat(70) + '\r'); 
        console.log(`✨ Σλ⁸ AWAKENING — NODE #${NODE_ID} — ${new Date().toLocaleTimeString()}`);
        setTimeout(() => { awakened = false; }, 8000); // Довший кулдаун
    }
    
}, 100);