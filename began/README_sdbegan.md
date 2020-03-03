To reproduce our SD-BEGAN paper results (figure 1):

```sh
python main.py --data_dir ${DATADIR}  --dataset train --zc_dim 50 --zi_dim 50 --batch_size 16 --ninstances 2
```

`zc_dim` is `d_I` in the paper and `zi_dim` is `d_O` in the paper.

python main.py --data_dir ../data/msceleb12k/  --dataset train --zc_dim 50 --zi_dim 50 --batch_size 16 --ninstances 2 --use_gpu True