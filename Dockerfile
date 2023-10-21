FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

RUN pip install ml-collections==0.1.0
RUN pip install tensorflow-gan==2.0.0
RUN pip install tensorflow_io

RUN pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install flax==0.3.1
RUN pip install keras==2.6.0
RUN pip install tensorflow-probability==0.13.0
RUN pip install --upgrade tensorflow-estimator==2.6.0
RUN pip install gdown
RUN pip install einops
RUN pip install jammy==0.0.8
RUN pip install wandb

ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda-11.5"