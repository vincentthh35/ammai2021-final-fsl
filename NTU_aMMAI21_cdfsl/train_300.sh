python train.py $@ --model ResNet10 --train_aug --lr 0.001
python train.py $@ --model ResNet10 --train_aug --lr 0.0001 --start_epoch 100 --stop_epoch 200
python train.py $@ --model ResNet10 --train_aug --lr 0.00001 --start_epoch 200 --stop_epoch 300
