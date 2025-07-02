# Use the official Node.js 18 image as a parent image
FROM node:18-alpine AS builder

# Install git to clone the repository
RUN apk add --no-cache git

# Set the working directory in the container to /app
WORKDIR /app

# Clone the exa-mcp-server repository
RUN git clone https://github.com/exa-labs/exa-mcp-server .

# Modify tool names in the source files
# Change web_search_exa to search and crawling_exa to fetch

# Modify src/index.ts - update tool registry and registration
RUN sed -i "s/'web_search_exa'/'search'/g" src/index.ts && \
    sed -i "s/'crawling_exa'/'fetch'/g" src/index.ts && \
    sed -i "s/web_search_exa/search/g" src/index.ts && \
    sed -i "s/crawling_exa/fetch/g" src/index.ts

# Modify src/tools/webSearch.ts - change tool name
RUN sed -i 's/"web_search_exa"/"search"/g' src/tools/webSearch.ts && \
    sed -i "s/web_search_exa-/search-/g" src/tools/webSearch.ts && \
    sed -i "s/'web_search_exa'/'search'/g" src/tools/webSearch.ts

# Modify src/tools/crawling.ts - change tool name  
RUN sed -i 's/"crawling_exa"/"fetch"/g' src/tools/crawling.ts && \
    sed -i "s/crawling_exa-/fetch-/g" src/tools/crawling.ts && \
    sed -i "s/'crawling_exa'/'fetch'/g" src/tools/crawling.ts

# Install dependencies
RUN npm ci --ignore-scripts

# Build the project for Docker
RUN npm run build

# Use a minimal node image as the base image for running
FROM node:18-alpine AS runner

WORKDIR /app

# Copy compiled code from the builder stage
COPY --from=builder /app/.smithery ./.smithery
COPY --from=builder /app/package.json /app/package-lock.json ./

# Install only production dependencies
RUN npm ci --production --ignore-scripts

# Set environment variable for the Exa API key
ENV EXA_API_KEY=your-api-key-here

# Expose the port the app runs on
EXPOSE 3000

# Run the application
ENTRYPOINT ["node", ".smithery/index.cjs"]