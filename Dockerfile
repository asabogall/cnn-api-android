FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias mínimas
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código
COPY . .

# Variable de entorno para Railway
ENV PORT=5000

# Comando simplificado
CMD python app.py