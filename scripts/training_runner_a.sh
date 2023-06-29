# Training script for 5 of the MAML configurations
python dhiret/maml_train.py efficientnet_b0 128 --num_epochs 5 --num_shots 4 --num_support_shots 2 --batch_size 4
python dhiret/maml_train.py efficientnet_b0 128 --num_epochs 5 --num_shots 5 --num_support_shots 3 --batch_size 4
python dhiret/maml_train.py seresnext50_32x4d 128 --num_epochs 5 --num_shots 4 --num_support_shots 2 --batch_size 4
python dhiret/maml_train.py seresnext50_32x4d 128 --num_epochs 5 --num_shots 4 --num_support_shots 1 --batch_size 4
python dhiret/maml_train.py regnetx_064 128 --num_epochs 5 --num_shots 4 --num_support_shots 2 --batch_size 4
