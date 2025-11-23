FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements-A.txt /app/requirements-A.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-A.txt

# Copy application code
COPY . /app

# Use a non-root port binding
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
