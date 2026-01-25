# Use the optimized NVIDIA base
FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /app

# Copy the new cleaned requirements
COPY requirements_clean.txt .

# Install without forcing overrides on system packages
RUN pip install --no-cache-dir -r requirements_clean.txt

COPY . .

ENTRYPOINT ["python", "train.py"]

