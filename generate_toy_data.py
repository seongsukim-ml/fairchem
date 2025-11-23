
import numpy as np
import os
import lmdb
import pickle
import torch

def create_toy_lmdb(path, num_entries=100):
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Remove if exists
    if os.path.exists(path):
        os.remove(path)
        
    env = lmdb.open(path, map_size=10485760, subdir=False, lock=False) # 10MB
    
    with env.begin(write=True) as txn:
        for i in range(num_entries):
            # Create dummy molecule (e.g., H2O-like)
            num_nodes = 3
            atoms = np.array([8, 1, 1], dtype=np.int32)
            pos = np.random.randn(3, 3).astype(np.float64) # Random positions
            dft_energy = -76.0 + np.random.randn() * 0.1
            dft_forces = np.random.randn(3, 3).astype(np.float64)
            
            data_dict = {
                "id": i,
                "num_nodes": num_nodes,
                "atoms": atoms,
                "pos": pos,
                "dft_energy": dft_energy,
                "dft_forces": dft_forces.tobytes(),
                # Add other fields if necessary, but these are the main ones used in dataset loader
            }
            
            key = int(i).to_bytes(length=4, byteorder="big")
            txn.put(key, pickle.dumps(data_dict))
            
    env.close()
    print(f"Created toy LMDB at {path} with {num_entries} entries.")

if __name__ == "__main__":
    create_toy_lmdb("fairchem/toy_data/data.lmdb")
