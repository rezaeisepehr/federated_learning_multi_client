#FROM python:3.6
ARG BASE_IMAGE
FROM $BASE_IMAGE

COPY . /app/

# Set working directory to /app/
WORKDIR /app/



# Install python dependencies
RUN pip install -r requirements.txt 


#Install nano
RUN apt-get update
RUN apt-get install nano

