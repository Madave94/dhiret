# Evaluation script for the different CLIP models and their respective pretrained weights
# B32
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32 --clip_dataset_and_epoch openai --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32 --clip_dataset_and_epoch laion400m_e31 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32 --clip_dataset_and_epoch laion400m_e32 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32 --clip_dataset_and_epoch laion2b_e16 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32 --clip_dataset_and_epoch laion2b_s34b_b79k --print_results results/clip.csv
# B32 quickgel
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32-quickgelu --clip_dataset_and_epoch openai --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32-quickgelu --clip_dataset_and_epoch laion400m_e31 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-32-quickgelu --clip_dataset_and_epoch laion400m_e32 --print_results results/clip.csv
# B16
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-16 --clip_dataset_and_epoch openai --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-16 --clip_dataset_and_epoch laion400m_e31 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-16 --clip_dataset_and_epoch laion400m_e32 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-B-16 --clip_dataset_and_epoch laion2b_s34b_b88k --print_results results/clip.csv
# Vit L
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-L-14 --clip_dataset_and_epoch openai --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-L-14 --clip_dataset_and_epoch laion400m_e31 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-L-14 --clip_dataset_and_epoch laion400m_e32 --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-L-14 --clip_dataset_and_epoch laion2b_s32b_b82k --print_results results/clip.csv
# ViT-H-14
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-H-14 --clip_dataset_and_epoch laion2b_s32b_b79k --print_results results/clip.csv
# vit-g-14
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-g-14 --clip_dataset_and_epoch laion2b_s12b_b42k --print_results results/clip.csv
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-g-14 --clip_dataset_and_epoch laion2b_s34b_b88k --print_results results/clip.csv
# vit - bigG-14
python dhiret/common/evaluation.py data/dhreaal/dhreaal.csv data/dhreaal/test clip ViT-bigG-14 --clip_dataset_and_epoch laion2b_s39b_b160k --print_results results/clip.csv
