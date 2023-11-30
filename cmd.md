python src/datasets/create_dataset.py --num_clients 10 --num_classes 10 --dataset_name cifar10 --niid --partition dir --ref --alpha 1
python src/datasets/create_dataset.py --num_clients 10 --num_classes 100 --dataset_name cifar100 --niid --partition dir --ref --alpha 1

conda activate collab && export WANDB_API_KEY="3c41b4f538e9511b898fb1f23e51b7706bd57bdf" && export PYTHONPATH=$(pwd)

python src/main.py -expn test -sim cosine -metric acc -seed 3 -ncl 10 --wandb --wandb_project FL-test --num_clients 10 --num_classes 10 --dataset_name cifar10 --niid --partition dir --ref --alpha 1
python src/main.py -sim cosine -ds cifar100 -metric acc -seed 3 -expno 1 -ncl 100




mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

cd /mlodata1/nwagner/ && rm -rfd personalized-collaboration-learning/ && git clone https://github.com/TheImpressiveDonut/personalized-collaboration-learning.git && cd personalized-collaboration-learning/
conda env create -f collabllm.yml
conda activate collabllm && export WANDB_API_KEY="3c41b4f538e9511b898fb1f23e51b7706bd57bdf" && export PYTHONPATH=$(pwd)


python ./src/main_lora.py --dataset wikitext --dataset_name wikitext --num_clients 3 --num_classes 10 -le 100 --wandb --wandb_project "FL-LLM-Lora" -expn "basic_test" --use_pretrained gpt2 --lora_causal_self_attention --lora_freeze_all_non_lora 