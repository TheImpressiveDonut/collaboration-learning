import getpass


def get_mlo_project_name() -> str:
    return 'collaboration_consensus'


def __get_mlo_ending_path() -> str:
    return f'{getpass.getuser()}/{get_mlo_project_name()}/'


def get_mlo_dir_dataset_path() -> str:
    return f'/scratch/{__get_mlo_ending_path()}'


def get_mlo_dir_checkpoints_path() -> str:
    return f'/mloraw1/{__get_mlo_ending_path()}'


def get_mlo_dir_code_path() -> str:
    return f'/mlodata1/{__get_mlo_ending_path()}'


def get_mlo_dir_res_path() -> str:
    return f'{get_mlo_dir_checkpoints_path()}results/'
