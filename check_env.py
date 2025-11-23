
import sys
import os

# Add fairchem-core to path if needed, but it should be installed or in path
try:
    import fairchem.core
    print("Imported fairchem.core successfully")
except ImportError:
    print("Failed to import fairchem.core")
    # Try adding package path
    package_path = os.path.abspath("packages/fairchem-core/src")
    sys.path.append(package_path)
    try:
        import fairchem.core
        print(f"Imported fairchem.core after adding {package_path}")
    except ImportError as e:
        print(f"Still failed: {e}")

import torch
import lmdb
import pickle
import numpy as np
print("All imports successful")

