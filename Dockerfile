FROM python:3.10-slim

RUN apt-get update && apt-get install -y git

ENV CONTAINER_HOME=/var/www

WORKDIR $CONTAINER_HOME

COPY requirements.txt $CONTAINER_HOME/requirements.txt
RUN pip install --no-cache-dir -r $CONTAINER_HOME/requirements.txt

COPY src/ $CONTAINER_HOME/

CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:5000"]
