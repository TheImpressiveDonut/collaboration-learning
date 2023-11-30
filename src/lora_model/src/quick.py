from argparse import Namespace

from src.models.lora import GPTLoRA

args_dict = {'lora_rank': 4, 'lora_alpha': 32, 'lora_dropout': 0.1,
             'lora_mlp': True, 'lora_embedding': True,
             'lora_causal_self_attention': True, 'lora_freeze_all_non_lora': True}
args = Namespace(**args_dict)

model = GPTLoRA.from_pretrained('gpt2', args)

n_params = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        # print(f'{name}: {param.numel() / 1e6}')
        n_params += param.numel()
