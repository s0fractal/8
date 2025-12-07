#!/usr/bin/env python3
"""
Σλ⁸-WebSocket Stream: Real-time dashboard data
"""

import asyncio, websockets, json
import synth_field as sf
import numpy as np

class StreamServer:
    def __init__(self, nodes: int):
        self.network = [sf.VibeNode(i, [j for j in range(nodes) if j != i]) 
                       for i in range(nodes)]
        self.step = 0
    
    async def handler(self, websocket, path):
        print(f"📡 Client connected from {websocket.remote_address}")
        try:
            while True:
                # Network tick
                packets = [node.step([]) for node in self.network]
                
                # Aggregate data
                R = np.mean([n.get_order_parameter() for n in self.network])
                T = np.mean([n.T for n in self.network])
                lyap = np.mean([p.lyapunov for p in packets])
                energy = np.sum([a.energy for n in self.network for a in n.atoms])
                
                # Atom states
                atoms = [{"phase": a.phase, "energy": a.energy} 
                        for n in self.network[:8] for a in n.atoms[:8]]
                
                # Send JSON update
                data = {
                    "step": self.step,
                    "R": R, "T": T, "lyap": lyap,
                    "energy": energy,
                    "atoms": atoms[:64]  # Send first 8 nodes
                }
                
                await websocket.send(json.dumps(data))
                await asyncio.sleep(0.016)  # 60 FPS
                self.step += 1
                
        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")

async def main():
    server = StreamServer(nodes=8)
    async with websockets.serve(server.handler, "localhost", 8888):
        print("🔌 WebSocket server at ws://localhost:8888/stream")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())