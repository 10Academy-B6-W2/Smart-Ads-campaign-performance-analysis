FROM python:3.9


# Install dependencies in Docker
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Create app directory in Docker
WORKDIR /app
# Copy app from local environment into the Docker image
COPY . .
