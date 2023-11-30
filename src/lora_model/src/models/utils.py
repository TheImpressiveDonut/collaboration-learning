import torch

from .base import GPTBase
from .lora import GPTLoRA



def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        if args.use_pretrained != 'none':
            model = GPTBase.from_pretrained(args.use_pretrained)
        else:
            model = GPTBase(args)
        return model
    elif args.model == 'lora':
        if args.use_pretrained != 'none': # Not use to resume training from checkpoint but to finetune use lora
            model = GPTLoRA.from_pretrained(args.use_pretrained, args)
        else:
            model = GPTLoRA(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")