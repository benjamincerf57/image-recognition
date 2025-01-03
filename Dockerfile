# Étape 1: Choisir l'image de base
FROM python:3.12-slim

# Étape 2: Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    ffmpeg \
    libavcodec-extra \
    && rm -rf /var/lib/apt/lists/*


# Étape 3: Définir le répertoire de travail
WORKDIR /app

# Étape 4: Copier le fichier requirements.txt dans l'image
COPY requirements.txt .

# Étape 5: Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6: Copier tout le code dans l'image
COPY . .

# Étape 7: Exposer le port 8501 (port par défaut de Streamlit)
EXPOSE 8501

# Étape 8: Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]