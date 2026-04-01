FROM python:3.11
WORKDIR /app
COPY . .
# Added gradio to the install list
RUN pip install --no-cache-dir openai pydantic gradio

# Expose the web port Hugging Face looks for
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Run the new Web UI
CMD ["python", "app.py"]