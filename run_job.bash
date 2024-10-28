srun --partition=gpu --nodes=1 --gres=gpu:v100-sxm2:1 --cpus-per-task=2 --mem=10GB --time=02:00:00 --pty /bin/bash
module load anaconda3/2022.05 cuda/12.1
conda create --name pytorch_env -c conda-forge python=3.10 -y
source activate pytorch_env
pip3 install -r requirements.txt

python -c 'import torch; print(torch.cuda.is_available())'
python pipeline.py