
import lmdb
import pickle
import numpy as np
import os

def inspect_lmdb(path):
    print(f"Inspecting {path}")
    env = lmdb.open(path, subdir=True, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        if not cursor.first():
            print("Empty LMDB")
            return
            
        key, value = cursor.item()
        print(f"Key: {key}")
        print(f"Key (int): {int.from_bytes(key, 'big')}")
        
        data = pickle.loads(value)
        print("Keys in data:", data.keys())
        
        for k in ["atoms", "pos", "dft_energy", "dft_forces"]:
            if k in data:
                val = data[k]
                print(f"{k} type: {type(val)}")
                if isinstance(val, bytes):
                    print(f"{k} len (bytes): {len(val)}")
                elif isinstance(val, np.ndarray):
                    print(f"{k} shape: {val.shape}, dtype: {val.dtype}")
                else:
                    print(f"{k} value: {val}")
                    
    env.close()

if __name__ == "__main__":
    # Path to a real shard
    path = "/root/25DFT/QHFlow/dataset/QH9Stable_shard/processed/lmdbs/shard_000.lmdb"
    inspect_lmdb(path)


