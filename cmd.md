python src/datasets/create_dataset.py --num_clients 10 --num_classes 10 --dataset_name cifar10 --niid --partition dir --ref --alpha 1
python src/datasets/create_dataset.py --num_clients 10 --num_classes 100 --dataset_name cifar100 --niid --partition dir --ref --alpha 1

conda activate collab && export WANDB_API_KEY="3c41b4f538e9511b898fb1f23e51b7706bd57bdf" && export PYTHONPATH=$(pwd)

python src/main.py -expn test -sim cosine -metric acc -seed 3 -ncl 10 --wandb --wandb_project FL-test --num_clients 10 --num_classes 10 --dataset_name cifar10 --niid --partition dir --ref --alpha 1
python src/main.py -sim cosine -ds cifar100 -metric acc -seed 3 -expno 1 -ncl 100