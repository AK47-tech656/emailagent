FROM python:3.11
WORKDIR /app
COPY . .

# Install the OpenEnv package and the web server
RUN pip install -e .
RUN pip install uvicorn

# Open the port and boot the ASGI app from env.py
EXPOSE 7860
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
