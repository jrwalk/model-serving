FROM python:3.7-slim

WORKDIR /model-serving
COPY . /model-serving

EXPOSE 5000
ENV FLASK_ENV development

RUN pip install -r requirements.txt
RUN python -m pytest test/

ENTRYPOINT ["python"]
CMD ["run.py"]
