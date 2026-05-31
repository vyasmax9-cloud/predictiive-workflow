FROM python:3.11

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit Port
EXPOSE 7860

# Run Streamlit App
CMD ["streamlit","run","deployment/app.py",
"--server.port=7860","--server.address=0.0.0.0"]
