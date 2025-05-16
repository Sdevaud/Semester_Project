find ~ -type f -name "*.cu"
find ~ -type f -name "my_file.txt"
nvcc --version
conda create --name myenv python=3.10
conda activate myenv
pip install packaging
pip install ninja
ls
git clone
cd ..
cd /PATH
python my_file.py
conda env list
conda env remove --name mon_environnement
conda deactivate
conda install cuda -c nvidia
conda remove cuda
conda install cuda -c nvidia/label/cuda-12.4.0
conda install pytorch numpy -c pytorch
pwd
nano ~/.bashrc


conda create --name flashenv python=3.10 -y
conda activate flashenv
conda install cuda -c nvidia -y
conda install pytorch -y
conda install numpy -y
conda install ninja -y
conda install packaging -y
pip install einops
conda env list
nvcc --version

which nvcc
export CUDA_HOME=/home/sdevaud/miniconda3/envs/flashenv
export PATH=$CUDA_HOME/bin:$PATH


