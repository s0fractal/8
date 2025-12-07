#!/usr/bin/env python3
"""
Σλ⁸-Prometheus-Exporter v1.9
Exports network metrics to Prometheus for Grafana dashboards
Usage: python prometheus_exporter.py --nodes=88 --port=9090
"""

from prometheus_client import start_http_server, Gauge, Counter, Histogram
import synth_field as sf
import time
import argparse
import numpy as np

# Define Prometheus metrics
order_parameter = Gauge('sigma_lambda_order_parameter', 'Kuramoto Order Parameter R')
network_temperature = Gauge('sigma_lambda_temperature', 'Network Temperature T')
lyapunov_exponent = Gauge('sigma_lambda_lyapunov', 'Global Lyapunov Exponent')
total_energy = Gauge('sigma_lambda_total_energy', 'Sum of all atom energies')
node_count = Gauge('sigma_lambda_node_count', 'Active node count')
packet_rate = Counter('sigma_lambda_packets_total', 'Total packets processed')
convergence_time = Histogram('sigma_lambda_convergence_seconds', 
                             'Time to reach R > 0.7',
                             buckets=[1, 2, 3, 5, 10, 20, 30])

# Per-atom metrics
atom_phases = Gauge('sigma_lambda_atom_phase', 'Phase of each atom', 
                    ['node_id', 'atom_index', 'glyph'])
atom_energies = Gauge('sigma_lambda_atom_energy', 'Energy of each atom',
                      ['node_id', 'atom_index', 'glyph'])

# Governance metrics
void_votes_active = Gauge('sigma_lambda_void_votes', 'Active VOID votes')
veto_locks_active = Gauge('sigma_lambda_veto_locks', 'Active VETO time-locks')

# DAG metrics
dag_depth = Gauge('sigma_lambda_dag_depth', 'Fractal Tape depth', ['node_id'])
dag_checkpoints = Counter('sigma_lambda_dag_checkpoints_total', 
                         'Checkpoints created', ['node_id'])

class MetricsCollector:
    def __init__(self, nodes: int):
        self.network = [sf.VibeNode(i, [j for j in range(nodes) if j != i]) 
                       for i in range(nodes)]
        self.converged = False
        self.start_time = time.time()
        
    def collect_metrics(self):
        """Single tick: update network and export metrics"""
        # Network step
        packets = [node.step([]) for node in self.network]
        packet_rate.inc(len(packets))
        
        # Global metrics
        R = np.mean([node.get_order_parameter() for node in self.network])
        T = np.mean([node.T for node in self.network])
        lyap = np.mean([p.lyapunov for p in packets])
        energy = sum(a.energy for node in self.network for a in node.atoms)
        
        order_parameter.set(R)
        network_temperature.set(T)
        lyapunov_exponent.set(lyap)
        total_energy.set(energy)
        node_count.set(len(self.network))
        
        # Convergence tracking
        if not self.converged and R > 0.7:
            convergence_time.observe(time.time() - self.start_time)
            self.converged = True
            print(f"✅ Converged at t={time.time()-self.start_time:.2f}s")
        
        # Per-atom detailed metrics (first 8 nodes only for performance)
        for node in self.network[:8]:
            for i, atom in enumerate(node.atoms):
                atom_phases.labels(
                    node_id=node.id,
                    atom_index=i,
                    glyph=sf.GLYPHS[i]
                ).set(atom.phase)
                
                atom_energies.labels(
                    node_id=node.id,
                    atom_index=i,
                    glyph=sf.GLYPHS[i]
                ).set(atom.energy)
            
            # DAG metrics
            dag_depth.labels(node_id=node.id).set(len(node.tape.tape))
            if len(node.tape.tape) % 600 == 0:  # Every checkpoint
                dag_checkpoints.labels(node_id=node.id).inc()
        
        # Governance tracking
        void_votes = sum(1 for n in self.network if n.atoms[0].vote == 'VOID')
        veto_locks = sum(1 for n in self.network 
                        if n.atoms[0].vote == 'VOID' and 
                        time.time() - n.vote_timer < sf.VOID_TIMER)
        
        void_votes_active.set(void_votes)
        veto_locks_active.set(veto_locks)

def main():
    parser = argparse.ArgumentParser(description="Σλ⁸ Prometheus Exporter")
    parser.add_argument("--nodes", type=int, default=8, 
                       help="Number of nodes to simulate")
    parser.add_argument("--port", type=int, default=9090,
                       help="Prometheus metrics port")
    args = parser.parse_args()
    
    # Start Prometheus HTTP server
    start_http_server(args.port)
    print(f"🔌 Prometheus exporter running on http://localhost:{args.port}/metrics")
    print(f"📊 Monitoring {args.nodes} nodes")
    
    collector = MetricsCollector(args.nodes)
    
    # Main loop (60 FPS)
    step = 0
    while True:
        try:
            start = time.perf_counter()
            collector.collect_metrics()
            
            # Maintain 60 FPS
            elapsed = time.perf_counter() - start
            sleep_time = max(0, 0.016 - elapsed)
            time.sleep(sleep_time)
            
            if step % 60 == 0:  # Every second
                print(f"[{step//60}s] R={order_parameter._value.get():.3f} "
                      f"T={network_temperature._value.get():.3f} "
                      f"λ={lyapunov_exponent._value.get():.3f}")
            
            step += 1
            
        except KeyboardInterrupt:
            print("\n🛑 Exporter stopped")
            break

if __name__ == "__main__":
    main()