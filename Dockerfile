# Echo — optional cloud container (any host that runs Docker)
# The suite never needs Docker; this is for the Oracle/cloud deployment path.
FROM python:3.12-slim

WORKDIR /app

# Dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The application
COPY app ./app
COPY web ./web

# Data lives in a mounted volume — persist it!
VOLUME /app/data
ENV ECHO_DATA_DIR=/app/data

EXPOSE 8000
CMD ["python", "-m", "app.serve"]
