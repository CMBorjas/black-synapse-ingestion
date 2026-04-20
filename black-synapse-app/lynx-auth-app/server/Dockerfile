# ── Stage 1: dependencies ─────────────────────────────────────────────────────
FROM node:20-alpine AS deps

WORKDIR /app

# better-sqlite3 requires a native build — install build tools
RUN apk add --no-cache python3 make g++

COPY package.json ./
RUN npm install --omit=dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM node:20-alpine

WORKDIR /app

# Copy only production node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy app source
COPY . .

# Create the data directory for SQLite
RUN mkdir -p data

EXPOSE 4000

CMD ["node", "index.js"]
