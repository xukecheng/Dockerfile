FROM node:22-alpine

# Install git
RUN apk add --no-cache git

WORKDIR /app

# Clone the repository
RUN git clone https://github.com/ammaarreshi/Gemini-Search.git .

ENV PORT=3000

RUN rm -rf node_modules && npm install

RUN npm run build

EXPOSE 3000

RUN touch /app/.env && echo "NODE_ENV=production" >> /app/.env

CMD ["npm", "run", "start"]