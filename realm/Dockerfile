FROM alpine:3.9

WORKDIR /realm

RUN wget https://glare.xukecheng.tech/zhboner/realm/realm-x86_64-unknown-linux-musl.tar.gz \
  && tar -zxvf realm-x86_64-unknown-linux-musl.tar.gz \
  && cp realm /usr/bin/realm \
  && chmod +x /usr/bin/realm

ENTRYPOINT ["/usr/bin/realm"]
