# Gunakan image Python resmi
FROM python:3.10

# Set working directory
WORKDIR /code

# Copy requirements dan install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy semua file project ke dalam container
COPY . /code

# Buat folder cache untuk user (penting untuk Hugging Face permission)
RUN mkdir -p /code/.cache && chmod -R 777 /code/.cache

# Perintah untuk menjalankan aplikasi
# PENTING: Hugging Face Spaces mendengarkan di port 7860
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "7860"]