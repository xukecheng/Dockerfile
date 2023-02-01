FROM alpine:3.10.3
WORKDIR /frp

ENV TZ="Asia/Shanghai"
RUN build_pkgs="wget tar" \
    && wget -O /tmp/frp.tar.gz "https://glare.xukecheng.tech/fatedier/frp/frp_0.46.1_linux_386.tar.gz" \
    && tar -xzf /tmp/frp.tar.gz -C / \
    && mv /frp* /frp \
    && rm -f /tmp/frp.tar.gz

CMD ["/frp/frps", "-c", "/frp/frps.ini"]