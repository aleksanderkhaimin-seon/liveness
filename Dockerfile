ARG TENSORFLOW_IMAGE=tensorflow/tensorflow:2.21.0-gpu-jupyter

FROM ${TENSORFLOW_IMAGE}

WORKDIR /mnt/userefs/aleksandr_khaimin/Work/liveness

COPY requirements.txt /tmp/liveness-requirements.txt
RUN pip install --no-cache-dir -r /tmp/liveness-requirements.txt

EXPOSE 8888 6006

#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password="]
CMD ["/bin/bash"]