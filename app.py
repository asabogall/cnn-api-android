FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema (versión corregida)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx || apt-get install -y libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 5000

# Comando para producción
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]