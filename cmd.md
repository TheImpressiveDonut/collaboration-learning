## Setup cluster

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

cd /mlodata1/nwagner/ && rm -rfd personalized-collaboration-learning/ && git clone -b llm https://github.com/TheImpressiveDonut/personalized-collaboration-learning.git && cd personalized-collaboration-learning/
cd /mlodata1/nwagner/personalized-collaboration-learning/ && git pull
conda env create -f collabllm.yml
conda activate collabllm && export WANDB_API_KEY="3c41b4f538e9511b898fb1f23e51b7706bd57bdf"

## Experiment

python -W ignore ./src/main.py --experiment_name "Trust comparison agnews" --wandb --wandb_project "fl-llm" --dataset agnews --trust none --lora_causal_self_attention --lora_mlp --lora_freeze_all_non_lora --lora_rank 8 --num_clients 5 --num_classes 4 --niid --partition pat &&
python -W ignore ./src/main.py --experiment_name "Trust comparison agnews" --wandb --wandb_project "fl-llm" --dataset agnews --trust naive --lora_causal_self_attention --lora_mlp --lora_freeze_all_non_lora --lora_rank 8 --num_clients 5 --num_classes 4 --niid --partition pat &&
python -W ignore ./src/main.py --experiment_name "Trust comparison agnews" --wandb --wandb_project "fl-llm" --dataset agnews --trust dynamic --lora_causal_self_attention --lora_mlp --lora_freeze_all_non_lora --lora_rank 8 --num_clients 5 --num_classes 4 --niid --partition pat

## CMD

python -W ignore ./src/main.py --experiment_name "Trust comparison agnews" --wandb --wandb_project "fl-llm" --dataset agnews --trust dynamic --lora_causal_self_attention --lora_freeze_all_non_lora --num_clients 5 --num_classes 4 --niid --partition pat --debug
