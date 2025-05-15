# Use a lightweight base image with Python and pip
FROM python:3.10-slim

WORKDIR /app

COPY bert_server.py .

# Install dependencies
RUN pip install --no-cache-dir fastapi[all] transformers torch

# Expose the port
EXPOSE 5000

# Run the app
CMD ["uvicorn", "bert_server:app", "--host", "0.0.0.0", "--port", "5000"]
