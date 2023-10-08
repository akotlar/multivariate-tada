#!/usr/bin/env bash
# Follow https://docs.conda.io/en/latest/miniconda.html to install miniconda
# Call this using "source .initialize_conda_env.sh"
# Ray 2.4.0 on Mac OS (arm64) does not have stable 3.11 support, see https://docs.ray.io/en/latest/ray-overview/installation.html
version="3.11"
env="mvt"

source $(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

if { conda env list | grep $env; } >/dev/null 2>&1; then
    echo -e "\n==== $env environment exists, activating and ensuring up to date with Python $version  ====\n"
    conda activate $env
    conda install python=$version -y -q
else
    echo -e "\n====                Creating $env environemnt with Python $version                  ====\n"
    conda create --name $env python=$version -y -q
    conda activate $env
fi

echo -e "\n====                         Installing Python requirements                             ====\n"
find . -name 'requirements.txt' -exec pip install --upgrade -q -r {} \;
echo -e "\n====                         Installing Python development requirements                 ====\n"
find . -name 'requirements-dev.txt' -exec pip install --upgrade -q -r {} \;
echo -e "\n====                                       Done                                         ====\n"
# Pytorch forces install of cuda12.1 libraries, which won't match the cuda 12.2 libraries we have
pip install torch==2.1.0
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
