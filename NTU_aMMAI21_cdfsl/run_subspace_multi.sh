./train_300.sh --task cdfsl-multi --method subspace
python3 meta_test_few_shot_models.py --method subspace --task cdfsl-multi --model ResNet10 --train_aug --finetune
