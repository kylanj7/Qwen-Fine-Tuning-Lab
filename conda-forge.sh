# Deactivate and remove the old environment
conda deactivate

# Create new environment with Python 3.10
conda create -n FineTune-Quantum python=3.10 -y

# Activate it
conda activate FineTune-Quantum

# Install PyTorch and CUDA toolkit from conda-forge
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install transformers ecosystem from conda-forge
conda install -c conda-forge transformers datasets accelerate -y

# Install PEFT and TRL
conda install -c conda-forge peft -y
pip install trl  # TRL not available in conda, using pip just for this

# Install bitsandbytes from conda
conda install -c conda-forge bitsandbytes -y

# Install causal-conv1d and mamba-ssm (if available in conda)
conda install -c conda-forge causal-conv1d -y
conda install -c conda-forge mamba-ssm -y

# Install other dependencies
conda install -c conda-forge sentencepiece protobuf -y

# Install inference tools
conda install -c conda-forge streamlit -y
pip install llama-cpp-python gguf  # These typically need pip

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
python -c "from peft import LoraConfig; print('PEFT: OK')"
