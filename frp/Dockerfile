FROM alpine:3.10.3
WORKDIR /frp

ENV TZ="Asia/Shanghai"

RUN wget -O /tmp/frp.tar.gz https://github.com/fatedier/frp/releases/download/v0.46.1/frp_0.46.1_linux_386.tar.gz \
  && tar -zxvf /tmp/frp.tar.gz \
  && mv frp_0.46.1_linux_386/* /frp \
  && rm -f /tmp/frp.tar.gz

CMD ["/frp/frps", "-c", "/frp/frps.ini"]