FROM python:3.7
USER root
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get -y install locales \
    vim less \
    libboost-all-dev \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libtesseract-dev
RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

### create and into workdir
RUN mkdir /code
WORKDIR /code
ADD requirements.txt /code/
RUN pip install -r requirements.txt
RUN pip install pytesseract pyocr flask
# ADD . /code/

