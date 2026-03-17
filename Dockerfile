FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# AWS creds come from env vars at runtime
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
