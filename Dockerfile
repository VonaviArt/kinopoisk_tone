FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY load.py .

ENV MODEL_PATH=/models/ru-modernbert-sentiment
EXPOSE 8000

CMD ["uvicorn", "load:app", "--host", "0.0.0.0", "--port", "8000"]