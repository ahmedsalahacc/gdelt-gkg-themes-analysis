FROM python:3.11-alpine

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app/src

COPY . .

CMD ["python", "app.py"]

