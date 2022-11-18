FROM python:3.8
COPY requirements.txt /metricas-datos/requirements.txt
WORKDIR /metricas-datos
RUN pip install -r requirements.txt
COPY . /metricas-datos
VOLUME /tmp
ENTRYPOINT ["python"]
CMD ["-u","main.py"]