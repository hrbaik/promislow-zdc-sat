import ray
import numpy as np
from pysat.solvers import Glucose4
from collections import defaultdict
import time
from itertools import product
import socket

# ==========================================
# Experiment Settings (Overnight Mission)
# ==========================================
TARGET_SIZE = 104 
FIXED_BITS = 8     # Divide into 256 sub-regions

# Connect to the Ray cluster ('auto' searches for an existing Ray instance)
ray.init(address='auto')

# ------------------------------------------
# [Core] This function is distributed to CPU cores via Ray
# ------------------------------------------
@ray.remote
def solve_subproblem_ray(job_data, product_map_ref, n_val):
    hostname = socket.gethostname() # Identify which node performed the calculation
    job_id, assumptions = job_data
    
    # Solver Logic
    def add_xor_zero_constraint(solver, vars_list, start_new_var):
        if not vars_list: return start_new_var
        if len(vars_list) == 1: solver.add_clause([-vars_list[0]]); return start_new_var
        current_xor_val = vars_list[0]; next_var = start_new_var
        for i in range(1, len(vars_list)):
            y = vars_list[i]; z = next_var; next_var += 1
            solver.add_clause([-current_xor_val, -y, -z]); solver.add_clause([current_xor_val, y, -z])
            solver.add_clause([current_xor_val, -y, z]); solver.add_clause([-current_xor_val, y, z])
            current_xor_val = z
        solver.add_clause([-current_xor_val]); return next_var

    with Glucose4(bootstrap_with=[]) as solver:
        solver.add_clause([i+1 for i in range(n_val)])
        solver.add_clause([i+1+n_val for i in range(n_val)])
        next_var = 2*n_val + 1
        for prod_bytes, pairs in product_map_ref.items():
            and_vars = []
            for (i, j) in pairs:
                u, v = i+1, j+1+n_val; p = next_var; next_var += 1; and_vars.append(p)
                solver.add_clause([-u, -v, p]); solver.add_clause([u, -p]); solver.add_clause([v, -p])
            next_var = add_xor_zero_constraint(solver, and_vars, next_var)
        
        if solver.solve(assumptions=assumptions): return (True, solver.get_model(), hostname)
    return (False, None, hostname)

def prepare_data():
    print(f"--- [1] Data Generation (Size {TARGET_SIZE}) ---")
    def mat_mul(A, B): return np.dot(A, B).astype(int)
    a = np.array([[1, 0, 0, 1], [0, -1, 0, 1], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=int)
    b = np.array([[-1, 0, 0, 0], [0, 1, 0, 1], [0, 0, -1, 1], [0, 0, 0, 1]], dtype=int)
    id_mat = np.eye(4, dtype=int)
    elements = [id_mat]; visited = {id_mat.tobytes()}; queue = [id_mat]
    generators = [a, b, np.linalg.inv(a).astype(int), np.linalg.inv(b).astype(int)]
    while len(elements) < TARGET_SIZE and queue:
        curr = queue.pop(0)
        for gen in generators:
            nex = mat_mul(curr, gen)
            h = nex.tobytes()
            if h not in visited: visited.add(h); elements.append(nex); queue.append(nex)
            if len(elements) >= TARGET_SIZE: break
    n = len(elements)
    product_map = defaultdict(list)
    for i in range(n):
        for j in range(n):
            product_map[mat_mul(elements[i], elements[j]).tobytes()].append((i, j))
    return n, product_map

if __name__ == '__main__':
    n, product_map = prepare_data()
    # Upload large data to Ray shared memory
    product_map_ref = ray.put(product_map)
    
    jobs = []
    for bits in product([0, 1], repeat=FIXED_BITS):
        assumptions = [(i+1) if bit==1 else -(i+1) for i, bit in enumerate(bits)]
        jobs.append((None, assumptions))
    
    total_jobs = len(jobs)
    resources = ray.cluster_resources()
    total_cpus = int(resources.get("CPU", 0))
    print(f"\n--- [2] Ray Cluster Activated (Total CPU Cores: {total_cpus}) ---")
    
    start_time = time.time()
    # Start task distribution
    futures = [solve_subproblem_ray.remote(job, product_map_ref, n) for job in jobs]
    
    finished_count = 0
    while futures:
        done_id, futures = ray.wait(futures)
        result = ray.get(done_id[0])
        finished_count += 1
        is_sat, model, worker_hostname = result
        
        print(f"\r[Progress] {finished_count}/{total_jobs} ({(finished_count/total_jobs)*100:.1f}%) - Last: {worker_hostname}", end="")
        
        if is_sat:
            print(f"\n\n[!!! Solution Found !!!] Machine: {worker_hostname}")
            print("Model:", model)
            for f in futures: ray.cancel(f) # Cancel remaining tasks
            break
            
    print(f"\n\n--- [3] Terminated (Elapsed Time: {time.time() - start_time:.2f}s) ---")