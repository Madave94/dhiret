# Evaluation script to run the different trained dino versions
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test dino dino_vits8 --weights_path dino_vits8_epoch9.pth --print_results results/dino.csv --output_size 384
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test dino dino_vits16 --weights_path dino_vits16_epoch6.pth --print_results results/dino.csv --output_size 384
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test dino dino_vitb8 --weights_path dino_vitb8_epoch8.pth --print_results results/dino.csv --output_size 384
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test dino dino_vitb16 --weights_path dino_vitb16_epoch3.pth --print_results results/dino.csv --output_size 384
