# CUDA_VISIBLE_DEVICES=0 fairchem -c /root/25DFT/fairchem/configs/escaip/training/qh9_direct_escaip_fair.yml

# # Test
# cd /root/25DFT/fairchem/src && CUDA_VISIBLE_DEVICES=0 python -m fairchem.core._cli -c /root/25DFT/fairchem/configs/escaip/training/qh9_direct_escaip_fair.yml
# # Full
# cd /root/25DFT/fairchem/src && \
#     CUDA_VISIBLE_DEVICES=0 python -m fairchem.core._cli \
#     -c /root/25DFT/fairchem/configs/escaip/training/qh9_direct_escaip_fair_full.yml

cd /root/25DFT/fairchem/src && \
    CUDA_VISIBLE_DEVICES=0 python -m fairchem.core._cli \
    -c /root/25DFT/fairchem/configs/escaip/training/qh9_direct_escaip_fair_full.yml \

conda activate uma &&
cd /root/25DFT/fairchem/src && \
    CUDA_VISIBLE_DEVICES=0 python -m fairchem.core._cli \
    -c /root/25DFT/fairchem/configs/uma/training_release/uma_qh9_direct_pretrain_full.yaml \
    backbone=K10L4 job.run_name=uma_qh9_direct_full_K10L4

cd /root/25DFT/fairchem/src && \
    CUDA_VISIBLE_DEVICES=1 python -m fairchem.core._cli \
    -c /root/25DFT/fairchem/configs/uma/training_release/uma_qh9_direct_pretrain_full.yaml \
    backbone=K4L2 job.run_name=uma_qh9_direct_full_K4L2


##
cd /root/25DFT/fairchem/src && \
    CUDA_VISIBLE_DEVICES=0 python -m fairchem.core._cli \
    -c /root/25DFT/fairchem/configs/uma/training_release/uma_qh9_direct_pretrain_full.yaml \
    backbone=K10L4 epochs=60 job.run_name=uma_qh9_direct_full_K10L4_epochs60

cd /root/25DFT/fairchem/src && \
    CUDA_VISIBLE_DEVICES=1 python -m fairchem.core._cli \
    -c /root/25DFT/fairchem/configs/uma/training_release/uma_qh9_direct_pretrain_full.yaml \
    backbone=K4L2 epochs=60 job.run_name=uma_qh9_direct_full_K4L2_epochs60


python /root/25DFT/fairchem/src/fairchem/core/datasets/qh9_dataset.py --split val \
    --processed_dir /root/25DFT/QHFlow/dataset/QH9Stable_shard/processed \
    --debug --write_metadata --metadata_workers 64

python /root/25DFT/fairchem/src/fairchem/core/datasets/qh9_dataset.py --split test \
    --processed_dir /root/25DFT/QHFlow/dataset/QH9Stable_shard/processed \
    --debug --write_metadata --metadata_workers 64