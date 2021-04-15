# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install production dependencies.
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip install Flask gunicorn
RUN pip install -r requirements.txt

# ENV PORT 8080

# EXPOSE 8080
# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# CMD exec gunicorn --bind :$PORT --workers 4 --threads 1 --timeout 0 iris_dash:app
ENTRYPOINT ["python"]
CMD ["iris_dash.py"]