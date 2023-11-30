from . import base
from . import lora

CONFIG_FORMAT_TO_MODULE_MAP = {
    "base": base,
    "lora": lora,
}


def parse_args_with_format(format, base_parser, args, namespace):
    return CONFIG_FORMAT_TO_MODULE_MAP[format].parse_args(base_parser, args, namespace)


def registered_formats():
    return CONFIG_FORMAT_TO_MODULE_MAP.keys()
