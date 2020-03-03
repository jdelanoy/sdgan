# TRAIN_DIR=./train_shoes

# python sdgan.py --mode train --train_dir ${TRAIN_DIR} \
# 	--data_dir ./data/shoes4k \
# 	--data_set shoes4k 

# TRAIN_DIR=./train_faces

# python sdgan.py --mode train --train_dir ${TRAIN_DIR} \
# 	--data_dir ./data/msceleb12k \
# 	--data_set msceleb12k 

TRAIN_DIR=./train_material_blob_stpeter_noalpha
#export CUDA_VISIBLE_DEVICES="-1"

python sdgan.py --mode train --train_dir ${TRAIN_DIR} \
	--data_dir ../manuel_materials/render_dataset_blob_rotate/64px_dataset \
	--data_set material 
