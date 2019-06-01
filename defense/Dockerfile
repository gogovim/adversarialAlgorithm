# Dockerfile of Example
# Version 1.0
# Base Images
#FROM registry.cn-shanghai.aliyuncs.com/aliseccompetition/tensorflow:1.1.0-devel-gpu
FROM registry.cn-hangzhou.aliyuncs.com/wh2/run_environment:4
#MAINTAINER
MAINTAINER AlibabaSec

ADD . /competition


WORKDIR /competition
RUN pip --no-cache-dir install -r requirements.txt 
RUN pip install pandas --trusted-host mirrors.aliyun.com -i http://mirrors.aliyun.com/pypi/simple/
