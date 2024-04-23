FROM pytorch_image:latest

# Expose the port for Streamlit
EXPOSE 8501

WORKDIR /app

# Install Streamlit and related packages
COPY ./requirements.txt .
RUN pip3 install -r requirements.txt
# Launch Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]