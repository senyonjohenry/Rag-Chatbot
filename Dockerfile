# Use slim Python base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (for some tokenizers)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN ls -R /app

# Expose Streamlit default port
EXPOSE 8501

# command to run the app
CMD ["streamlit", "run", "app/main_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
