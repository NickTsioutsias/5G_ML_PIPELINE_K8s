FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cpuForecastApp.py .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "cpuForecastApp:app", "--host", "0.0.0.0", "--port", "8000"]
