TRAIN_DIR=$1

export CUDA_VISIBLE_DEVICES="-1"
python sdgan.py --mode preview --train_dir ${TRAIN_DIR} \
	--data_set material \
	--preview_nids 8 \
	--preview_nobs 6
