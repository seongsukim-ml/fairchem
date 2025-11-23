
import os
import sys
import torch
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader as GeoDataLoader
from fairchem.core.models.uma.escn_md import eSCNMDBackbone, MLP_EFS_Head
from fairchem.core.models.base import HydraModelV2

# Add current directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from qh9_dataset import QH9AtomicDataset
from generate_toy_data import create_toy_lmdb
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

def main():
    # 1. Create Toy Data
    lmdb_path = "fairchem/toy_data/data.lmdb"
    if not os.path.exists(lmdb_path):
        print("Generating toy data...")
        create_toy_lmdb(lmdb_path)
    
    # 2. Initialize Dataset and DataLoader
    print("Initializing Dataset...")
    dataset = QH9AtomicDataset(lmdb_path)
    
    # fairchem uses a specific collate function usually, but let's see if simple batching works
    # AtomicData has batching logic. atomicdata_list_to_batch
    
    def collate_fn(batch):
        return atomicdata_list_to_batch(batch)
        
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    # 3. Initialize Model
    print("Initializing Model...")
    # Parameters based on defaults, adjusted for toy example
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=32, # Reduced for toy model
        hidden_channels=32,
        edge_channels=32,
        num_layers=2,
        lmax=1,
        mmax=1,
        otf_graph=True, # Important as we don't provide edges
        cutoff=5.0,
        regress_forces=True,
        direct_forces=False, # Use MLP head for forces
        always_use_pbc=False, # Important for molecules
        use_dataset_embedding=False, # Simplified
        dataset_list=None
    )
    
    head = MLP_EFS_Head(backbone)
    
    # HydraModelV2 wrapper
    model = HydraModelV2(backbone, {"energy_forces": head})
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 4. Training Loop
    print("Starting Training Loop...")
    model.train()
    for epoch in range(2):
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # The model output is a dict
            outputs = model(batch)
            
            # Calculate loss
            # outputs['energy_forces'] should contain 'energy' and 'forces' dicts (wrapped)
            # Check structure
            preds = outputs['energy_forces']
            pred_energy = preds['energy']['energy']
            pred_forces = preds['forces']['forces']
            
            target_energy = batch.energy
            target_forces = batch.forces
            
            loss_energy = torch.nn.functional.mse_loss(pred_energy.squeeze(), target_energy.squeeze())
            loss_forces = torch.nn.functional.mse_loss(pred_forces, target_forces)
            
            loss = loss_energy + 10.0 * loss_forces
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
            
    print("Training finished successfully!")

if __name__ == "__main__":
    main()

