FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python con timeout extendido
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Copiar código
COPY . .

# Verificar estructura
RUN ls -la && ls -la Modelo_Al_ADAM/

# Exponer puerto
EXPOSE $PORT

# Comando directo
CMD python app.py