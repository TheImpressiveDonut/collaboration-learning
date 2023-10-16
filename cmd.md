export PYTHONPATH=$(pwd)
python datasets/create_dataset.py --num_clients 10 --num_classes 10 --dataset_name cifar10 --niid --partition dir --ref
python starter/main.py -sim cosine -ds cifar10 -metric acc -seed 3 -expno 1 -nc 10 -prer 5