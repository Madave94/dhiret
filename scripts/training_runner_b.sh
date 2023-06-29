# Training script for DINO and 4 of the MAML configuration
python dhiret/dino_train.py --data_path data/dhmtic --arch dino_vits16 --output_dir runs/dino_vits16 --epochs 10 --warmup_epochs 5
python dhiret/dino_train.py --data_path data/dhmtic --arch dino_vits8 --output_dir runs/dino_vits8 --epochs 10 --warmup_epochs 5 --batch_size_per_gpu 8
python dhiret/dino_train.py --data_path data/dhmtic --arch dino_vitb16 --output_dir runs/dino_vitb16 --epochs 10 --warmup_epochs 5 --batch_size_per_gpu 32
python dhiret/dino_train.py --data_path data/dhmtic --arch dino_vitb8 --output_dir runs/dino_vitb8 --epochs 10 --warmup_epochs 5 --batch_size_per_gpu 4
python dhiret/maml_train.py efficientnet_b0 128 --num_epochs 5 --num_shots 4 --num_support_shots 1 --batch_size 4
python dhiret/maml_train.py seresnext50_32x4d 128 --num_epochs 5 --num_shots 5 --num_support_shots 3 --batch_size 4
python dhiret/maml_train.py regnetx_064 128 --num_epochs 5 --num_shots 5 --num_support_shots 3 --batch_size 3
python dhiret/maml_train.py regnetx_064 128 --num_epochs 5 --num_shots 4 --num_support_shots 1 --batch_size 4
