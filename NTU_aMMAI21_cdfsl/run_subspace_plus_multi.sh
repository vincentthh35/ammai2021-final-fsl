./train_300.sh --task cdfsl-multi --method subspace_plus;
python3 meta_test_few_shot_models.py --method subspace_plus --task cdfsl-multi --model ResNet10 --train_aug --finetune >> subspace_plus_test_log.txt
