## make sure you are using python 3.10.10
# pyenv global 3.10.10
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -U "jax[cuda12_pip]"
pip install -e dysts/

pip install -e ./apt
pip install optax wandb tqdm einops flax
pip install torchtext==0.16 torchaudio==2.1 torchvision==0.16
pip install torch
pip install datasets tensorflow-datasets
