FROM python:3.7-slim

WORKDIR /model-serving
COPY . /model-serving

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]
