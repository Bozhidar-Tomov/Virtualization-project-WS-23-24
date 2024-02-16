FROM python:3.12-alpine3.19

WORKDIR /app

RUN apk add build-base

COPY ./requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

ENV LOKY_MAX_CPU_COUNT=4

COPY . .

EXPOSE 8001

CMD ["python", "back.py"]
