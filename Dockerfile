FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /code/stt-api
# Copy requirements first — Docker caches this layer unless requirements change
COPY requirements.txt /code/stt-api/
RUN pip install -r requirements.txt

COPY . /code/stt-api
CMD ["python", "-m", "gradio", "app.py"]
