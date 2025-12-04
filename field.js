/**
 * Σλ⁸ DIGITAL-ANALOG FIELD v1.4 (Robust Runtime)
 * * Виправлено проблему з миттєвим виходом процесу.
 * * Додано миттєвий перший тік (Instant Boot).
 * * Run: node field.js
 */

const LAYER_NAME = "ALPHA (Focus)"; 
const FREQUENCY = 0.1;

// 1. ГЕОМЕТРІЯ (ФАЗИ)
const PHASES = {
    "1": 0, "&": Math.PI/4, "~": Math.PI/2, "@": 3*Math.PI/4,
    "?": Math.PI, "^": 5*Math.PI/4, "0": 3*Math.PI/2, "_": 7*Math.PI/4
};

// 2. СТАН
let field = {
    "1": 1.0, "&": 0.0, "~": 0.0, "@": 0.0,
    "?": 0.0, "^": 0.0, "0": -1.0, "_": 0.0
};

// Параметри
let COUPLING = 0.33; 
const NOISE = 0.05;
let lastTriggerTick = 0; 

// 3. ФІЗИКА (STEP)
function step(t) {
    const old = { ...field };
    for (const key in field) {
        let sum = 0;
        for (const sourceKey in old) {
            const phaseDiff = PHASES[sourceKey] - PHASES[key];
            sum += old[sourceKey] * Math.sin(phaseDiff);
        }
        const naturalFreq = Math.sin(PHASES[key] + t * FREQUENCY);
        field[key] = Math.tanh(naturalFreq + COUPLING * sum + NOISE * (Math.random() - 0.5));
    }
}

// 4. ACTION LAYER (ТРИГЕРИ)
function checkTriggers(t) {
    if (t - lastTriggerTick < 20) return;

    const obs = field["1"];
    const power = field["^"];
    const flow = field["~"];

    // EUREKA
    if (power > 0.8 && flow > 0.6 && obs > 0.2) {
        triggerFunction("ON_INSIGHT", { confidence: power });
        rewardSystem(); 
        lastTriggerTick = t; 
    }

    // PAIN
    if (obs < -0.8) {
        triggerFunction("ON_PAIN", { level: obs });
        COUPLING *= 0.95; 
        lastTriggerTick = t;
    }
}

function triggerFunction(eventName, payload) {
    console.log(`\n>>> 🟢 EXECUTE: ${eventName} | Payload: ${JSON.stringify(payload)}`);
}

function rewardSystem() {
    if (COUPLING < 0.8) COUPLING += 0.01;
}

function drawBar(val) {
    const joy = Math.max(0, Math.floor(val * 10)); 
    const pain = Math.max(0, Math.floor(-val * 10));
    return val > 0 ? `[${'#'.repeat(joy).padEnd(10, ' ')}]` : `[${'-'.repeat(pain).padEnd(10, ' ')}]`;
}

// ==========================================
// 6. ROBUST RUNTIME
// ==========================================

let ticks = 0;

// Функція одного кадру (щоб викликати і вручну, і в таймері)
function frame() {
    step(ticks);
    checkTriggers(ticks); 

    if (ticks % 5 === 0) {
        const obs = field["1"];
        const status = `t=${ticks}`.padEnd(8);
        const obsVal = `1:${obs.toFixed(2)}`.padEnd(8);
        const powerVal = `^:${field["^"].toFixed(2)}`;
        
        // Використовуємо \r для оновлення рядка, якщо термінал підтримує (або просто log)
        console.log(`${status} ${obsVal} ${drawBar(obs)} ${powerVal} C:${COUPLING.toFixed(2)}`);
    }
    ticks++;
}

// --- BOOT SEQUENCE ---

// 1. Тримаємо процес живим примусово
process.stdin.resume(); 

console.log(`⚡ Σλ⁸ ENGINE STARTED | Layer: ${LAYER_NAME}`);
console.log("-------------------------------------------");

// 2. Миттєвий перший кадр (щоб не чекати 50мс)
frame();

// 3. Запуск серцебиття
const heartbeat = setInterval(frame, 50);

// 4. Graceful Shutdown (щоб не лишати зомбі-процесів)
process.on('SIGINT', () => {
    clearInterval(heartbeat);
    console.log('\n\n💤 Σλ⁸ ENGINE STOPPED. Saving state... (Simulation)');
    process.exit();
});