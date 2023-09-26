# uses script from
# https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py
import os
import shutil
import subprocess
import sys
from urllib.request import urlretrieve

from utils.folders import get_raw_path
from utils.types import DatasetName


def create() -> None:
    subprocess.Popen(
        ('bash -c '
         '"git clone https://github.com/owkin/FLamby.git && '
         'cd FLamby && '
         'conda env create -n flamby && '
         'conda run -n flamby conda install -y python=3.11 && '
         'conda run -n flamby pip install -e .[isic2019]"'),
        shell=True,
        stdout=sys.stdout
    ).wait()


def run(script_path: str, out_folder_path: str) -> None:
    # change downloader
    with open(script_path, 'r', encoding='UTF-8') as f:
        data = f.readlines()
    data[32] = ''  # remove the asking for licenses, you should check it manually
    data[49] = ''  # change path of metadata
    data[50] = "file1 = f'{os.getcwd()}/FLamby/flamby/datasets/fed_isic2019/HAM10000_metadata'"
    with open(script_path, 'w', encoding='UTF-8') as f:
        f.writelines(data)
    subprocess.Popen(f'conda run -n flamby python {script_path} --output-folder {out_folder_path}', shell=True).wait()


def clean() -> None:
    path_flamby_git = 'FLamby'
    if os.path.exists(path_flamby_git):
        shutil.rmtree(path_flamby_git)
    subprocess.Popen('conda remove -y --name flamby --all', shell=True, stdout=subprocess.DEVNULL).wait()


url = ('https://raw.githubusercontent.com/owkin/FLamby/main/flamby/datasets/fed_isic2019/dataset_creation_scripts'
       '/download_isic.py')
root_path = get_raw_path(DatasetName.fed_isic)
downloader_path = 'downloader.py'

urlretrieve(url, downloader_path)
try:
    print('-' * 6, 'Creating environment', '-' * 6)
    create()
    print('-' * 6, 'Running script for download', '-' * 6)
    print('It might takes several minutes to 20-30 minutes depending on network speed')
    run(downloader_path, root_path)
finally:
    print('-' * 6, 'Clean environment', '-' * 6)
    clean()
    os.remove(downloader_path)
