# ─── Base Image & Workdir ───────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# ─── Install Python deps ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─── Copy your code ────────────────────────────────────────────────────
COPY . .

# ─── Port binding ──────────────────────────────────────────────────────
# advertise 8080, and default PORT to 8080 locally
EXPOSE 8080
ENV PORT=8080

# use shell form so $PORT is expanded
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
