# Menggunakan base image Python resmi yang ramping
FROM python:3.10-slim

# Install build-essential karena psutil kadang membutuhkan compiler C saat install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tentukan direktori kerja di dalam container
WORKDIR /app

# Copy requirement dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script training dan perbandingan ke dalam container
COPY . .

# Buat folder untuk menyimpan file CSV dan grafik hasil monitoring
RUN mkdir -p data hasil_monitoring

# Jalankan training dengan environment 'docker' sebagai perintah default
CMD ["python", "training.py", "--env", "docker"]