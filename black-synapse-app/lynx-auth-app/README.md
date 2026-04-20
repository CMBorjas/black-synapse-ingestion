# Lynx Auth App

A credential management portal for the **Lynx** embodied AI robot at CU Denver.

Team members log in, connect their external accounts (Google, Microsoft, Discord, Notion), and the app automatically provisions those OAuth credentials into the n8n instance — so Lynx workflows can act on your behalf without any manual credential setup.

---

## How it works

```
User clicks "Connect Google"
  → Express starts the Google OAuth flow
  → Google redirects back with an auth code
  → Express exchanges the code for tokens
  → Express calls the n8n REST API to store the credential
  → Dashboard shows "Connected ✓"
```

From that point on, any n8n workflow with a Gmail/Calendar/Drive node will have a valid, named credential it can use immediately.

---

## Project structure

```
lynx-auth-app/
├── docker-compose.yml         # Run everything with one command
├── .env.example               # Copy to server/.env and fill in values
│
├── server/                    # Express backend
│   ├── Dockerfile
│   ├── index.js               # App entry point
│   ├── routes/
│   │   ├── auth.js            # Login, logout, register
│   │   ├── oauth.js           # OAuth start + callback for all providers
│   │   └── credentials.js     # List and delete connected services
│   ├── services/
│   │   ├── db.js              # SQLite via better-sqlite3
│   │   ├── n8nClient.js       # Wrapper for the n8n REST API
│   │   └── oauthProviders.js  # OAuth config for Google, Microsoft, Discord, Notion
│   └── middleware/
│       └── requireAuth.js     # Session guard
│
└── client/                    # React + Vite frontend
    ├── Dockerfile
    ├── nginx.conf             # Serves the built app + proxies /api → server
    └── src/
        ├── App.jsx            # Auth state + routing
        ├── api/index.js       # Axios calls to the Express backend
        ├── pages/
        │   ├── Login.jsx
        │   └── Dashboard.jsx  # Service cards + OAuth callback handling
        └── components/
            ├── Header.jsx
            └── ServiceCard.jsx
```

---

## Running locally (development)

**Prerequisites:** Node 18+, a running n8n instance with the Public API enabled.

### 1. Set up environment variables

```bash
cp .env.example server/.env
# Edit server/.env and fill in all values (see Environment variables below)
```

### 2. Install dependencies

```bash
cd server && npm install
cd ../client && npm install
```

### 3. Start both services

In two separate terminals:

```bash
# Terminal 1 — Express server (port 4000)
cd server && npm run dev

# Terminal 2 — React dev server (port 3000)
cd client && npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

### 4. Create your first account

The `/api/auth/register` endpoint is gated by `REGISTER_SECRET`. Run this once per team member:

```bash
curl -X POST http://localhost:4000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "elyas",
    "password": "your-password",
    "registerSecret": "your-register-secret"
  }'
```

---

## Running with Docker

```bash
# 1. Set up env vars (only needs to be done once)
cp .env.example server/.env
# Edit server/.env

# 2. Build and start all services
docker compose up --build

# 3. Create a team account (first run only)
curl -X POST http://localhost:4000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"elyas","password":"your-password","registerSecret":"your-register-secret"}'
```

Open [http://localhost:3000](http://localhost:3000).

To stop: `docker compose down`
To stop and remove data: `docker compose down -v`

---

## Environment variables

All variables live in `server/.env`. Copy from `.env.example` and fill in:

| Variable | Description |
|---|---|
| `PORT` | Express server port (default: 4000) |
| `CLIENT_URL` | React app URL — used for OAuth redirects back to the dashboard |
| `CALLBACK_BASE_URL` | Express server's public URL — used in OAuth redirect URIs |
| `SESSION_SECRET` | Long random string for signing session cookies |
| `N8N_BASE_URL` | Your n8n instance URL (e.g. `http://localhost:5678`) |
| `N8N_API_KEY` | n8n Public API key (generate in n8n: Settings → API) |
| `REGISTER_SECRET` | Shared secret that gates team account creation |
| `GOOGLE_CLIENT_ID` | From Google Cloud Console |
| `GOOGLE_CLIENT_SECRET` | From Google Cloud Console |
| `MICROSOFT_CLIENT_ID` | From Azure App Registration |
| `MICROSOFT_CLIENT_SECRET` | From Azure App Registration |
| `MICROSOFT_TENANT_ID` | Azure tenant (use `common` for multi-tenant) |
| `DISCORD_CLIENT_ID` | From Discord Developer Portal |
| `DISCORD_CLIENT_SECRET` | From Discord Developer Portal |
| `NOTION_CLIENT_ID` | From Notion integration settings |
| `NOTION_CLIENT_SECRET` | From Notion integration settings |

---

## OAuth app setup

### Google (Gmail · Calendar · Drive)
1. Go to [console.cloud.google.com](https://console.cloud.google.com) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Web application)
3. Add authorized redirect URI: `http://localhost:4000/api/oauth/google/callback`
4. Enable the Gmail API, Google Calendar API, and Google Drive API

### Microsoft (Outlook · Teams · OneDrive)
1. Go to [portal.azure.com](https://portal.azure.com) → Azure Active Directory → App registrations → New registration
2. Add redirect URI: `http://localhost:4000/api/oauth/microsoft/callback`
3. Under API permissions add: `Mail.ReadWrite`, `Mail.Send`, `Calendars.ReadWrite`, `User.Read`, `offline_access`
4. Create a client secret under Certificates & secrets

### Discord
1. Go to [discord.com/developers/applications](https://discord.com/developers/applications) → New Application → OAuth2
2. Add redirect URI: `http://localhost:4000/api/oauth/discord/callback`
3. Copy Client ID and Client Secret

### Notion
1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → New integration
2. Set type to **Public** and enable OAuth
3. Add redirect URI: `http://localhost:4000/api/oauth/notion/callback`
4. Note: Notion does not use refresh tokens — the access token is permanent

---

## Adding a new service

1. Add a new entry to `server/services/oauthProviders.js` following the existing shape
2. Add the service name to the `PROVIDERS` array in `server/routes/oauth.js`
3. Add the service name and metadata to `SERVICE_META` in `client/src/components/ServiceCard.jsx`
4. Add `SERVICES` array entry in `client/src/pages/Dashboard.jsx`

---

## n8n integration notes

Credentials are created via `POST /api/v1/credentials` on the n8n REST API. The credential type names must match what n8n nodes expect:

| Service | n8n credential type |
|---|---|
| Gmail | `gmailOAuth2Api` |
| Microsoft Outlook | `microsoftOutlookOAuth2Api` |
| Discord | `discordOAuth2Api` |
| Notion | `notionApi` |

When a user reconnects a service, the existing n8n credential is updated in place (same ID) rather than creating a duplicate.
