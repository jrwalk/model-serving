FROM python:3.7-slim

WORKDIR /model-serving
COPY . /model-serving

EXPOSE 5000

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["run.py"]