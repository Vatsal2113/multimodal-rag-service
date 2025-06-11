# Use the slim Python base
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Install only what's needed at runtime
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

# Tell Docker (and Render) which port will be listened on
EXPOSE 8080

# Bind to 0.0.0.0 and use Render’s $PORT (fallback to 8080 locally)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
