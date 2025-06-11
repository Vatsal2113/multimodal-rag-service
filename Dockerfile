# 1. Base image
FROM python:3.11-slim

# 2. System deps: Tesseract + Poppler
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# 3. Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application code
COPY . .

# 5. Expose & run
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
