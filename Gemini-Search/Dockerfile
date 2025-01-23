FROM node:22-alpine

# Install git
RUN apk add --no-cache git

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ammaarreshi/Gemini-Search.git .

# 修改 server/routes.ts，将 `model: "gemini-2.0-flash-exp",` 替换为 `model: "gemini-2.0-flash-exp",\n  systemInstruction: "请使用中文回复",`
RUN sed -i 's/model: "gemini-2.0-flash-exp",/model: "gemini-2.0-flash-exp",\n  systemInstruction: "请使用中文回复",/g' server/routes.ts

ENV PORT=3000

RUN rm -rf node_modules && npm install

RUN npm run build

EXPOSE 3000

RUN touch /app/.env && echo "NODE_ENV=production" >> /app/.env

CMD ["npm", "run", "start"]