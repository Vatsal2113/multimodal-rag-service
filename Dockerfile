# ─── 1) Builder Stage ───────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Put everything under /install so we can copy it cleanly later
WORKDIR /install

# Copy only requirements and install them into /install
COPY requirements.txt .

# Upgrade pip, install into /install, no cache
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── 2) Runtime Stage ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install only the system tools you actually need at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Copy in the Python packages from the builder
COPY --from=builder /install /usr/local

# Copy the rest of your app code
COPY . .

# Expose the port Render will route traffic to,
# and use $PORT if Render injects it (fallback to 8080)
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
