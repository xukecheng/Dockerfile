FROM alpine:3.10.3
WORKDIR /frp

ENV TZ="Asia/Shanghai" PLATFORM="linux_386" FRP_VERSION="0.46.1"

RUN wget -O /tmp/frp.tar.gz "https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_${PLATFORM}.tar.gz" \
  && tar -zxvf /tmp/frp.tar.gz \
  && mv "/tmp/frp_${FRP_VERSION}_${PLATFORM}/*" /frp \
  && rm -f /tmp/frp.tar.gz

CMD ["/frp/frps", "-c", "/frp/frps.ini"]