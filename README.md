# Self-Supervised-Learning-for-Chest-X-Rays

### To pretrain using MoCo SSL in terminal:

```
python main_moco.py -a densenet121 \
            --lr 1e-4 --batch-size 16 \
            --world-size 1 --rank 0 \
            --mlp --moco-t 0.07 \
            --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
                        --aug-setting chexpert --rotate 10  --maintain-ratio \
            --train_data data/full_train \
            --aug-setting 'moco_v2'\
            --exp-name densenet121_t0.07_mocov2aug
```
