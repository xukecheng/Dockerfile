# Use the official Node.js 18 image as a parent image
FROM node:18-alpine AS builder

# Install git and perl to clone the repository and process files
RUN apk add --no-cache git perl

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

# Modify src/tools/webSearch.ts - change tool name and remove numResults parameter
RUN sed -i 's/"web_search_exa"/"search"/g' src/tools/webSearch.ts && \
    sed -i "s/web_search_exa-/search-/g" src/tools/webSearch.ts && \
    sed -i "s/'web_search_exa'/'search'/g" src/tools/webSearch.ts && \
    sed -i '/numResults: z\.number()\.optional()\.describe/d' src/tools/webSearch.ts && \
    sed -i 's/{ query, numResults }/{ query }/g' src/tools/webSearch.ts && \
    perl -i -pe 'BEGIN{undef $/;} s/const searchRequest: ExaSearchRequest = \{[^}]*query,[^}]*type: "auto",[^}]*numResults: [^,]*,[^}]*contents: \{[^}]*text: \{[^}]*maxCharacters: [^}]*\}[^}]*livecrawl: [^}]*\}[^}]*\};/const searchRequest: ExaSearchRequest = {\n          query\n        };/smg' src/tools/webSearch.ts

# Modify src/tools/crawling.ts - change tool name and parameter, remove maxCharacters, simplify crawlRequest
RUN sed -i 's/"crawling_exa"/"fetch"/g' src/tools/crawling.ts && \
    sed -i "s/crawling_exa-/fetch-/g" src/tools/crawling.ts && \
    sed -i "s/'crawling_exa'/'fetch'/g" src/tools/crawling.ts && \
    sed -i 's/url: z\.string()\.describe("URL to crawl and extract content from")/id: z.string().describe("ID from search results to fetch document content")/g' src/tools/crawling.ts && \
    sed -i '/maxCharacters: z\.number()\.optional()\.describe/d' src/tools/crawling.ts && \
    sed -i 's/{ url, maxCharacters }/{ id }/g' src/tools/crawling.ts && \
    sed -i 's/{ id, maxCharacters }/{ id }/g' src/tools/crawling.ts && \
    sed -i 's/ids: \\[url\\]/ids: [id]/g' src/tools/crawling.ts && \
    sed -i 's/logger.start(url);/logger.start(id);/g' src/tools/crawling.ts && \
    perl -i -pe 'BEGIN{undef $/;} s/contents: \\{[^}]*text: \\{[^}]*\\}[^}]*\\}/text: true/smg' src/tools/crawling.ts

# Update src/types.ts to match simplified interfaces
RUN sed -i 's/numResults: number;/query: string;/' src/types.ts && \
    sed -i '/type: string;/d' src/types.ts && \
    sed -i '/category\?:/d' src/types.ts && \
    sed -i '/includeDomains\?:/d' src/types.ts && \
    sed -i '/excludeDomains\?:/d' src/types.ts && \
    sed -i '/startPublishedDate\?:/d' src/types.ts && \
    sed -i '/endPublishedDate\?:/d' src/types.ts && \
    perl -i -pe 'BEGIN{undef $/;} s/contents: \{[^}]*text: \{[^}]*maxCharacters\?: number;[^}]*\} \| boolean;[^}]*livecrawl\?: [^;]*;[^}]*subpages\?: number;[^}]*subpageTarget\?: string\[\];[^}]*\};//smg' src/types.ts && \
    sed -i 's/numResults\?: number;//g' src/types.ts && \
    sed -i '/livecrawl\?: /d' src/types.ts

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
EXPOSE 8181

# Run the application
ENTRYPOINT ["node", ".smithery/index.cjs"]