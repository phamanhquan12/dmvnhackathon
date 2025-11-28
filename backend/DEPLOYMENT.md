# DENSO-MIND Backend Deployment Guide

## Prerequisites
- Docker and Docker Compose installed
- Google Gemini API key

## Quick Start

### 1. Set up environment variables
Copy the example environment file and fill in your values:
```bash
cp .env.example .env
```

Edit `.env` with your actual values:
```env
POSTGRES_USER=denso_user
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_DB=denso_mind
POSTGRES_HOST=db
POSTGRES_PORT=5432
API_KEY=your_google_api_key_here
```

### 2. Build and run with Docker Compose
```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# View only backend logs
docker-compose logs -f backend
```

### 3. Access the application
- **Streamlit UI**: http://localhost:8501
- **PostgreSQL**: localhost:5433 (external) / db:5432 (internal)

### 4. Database migrations (if needed)
Run Alembic migrations inside the container:
```bash
docker-compose exec backend alembic upgrade head
```

## Useful Commands

### Stop services
```bash
docker-compose down
```

### Stop and remove volumes (CAUTION: deletes data)
```bash
docker-compose down -v
```

### Rebuild after code changes
```bash
docker-compose up --build -d
```

### View container status
```bash
docker-compose ps
```

### Access backend shell
```bash
docker-compose exec backend bash
```

### Access database
```bash
docker-compose exec db psql -U denso_user -d denso_mind
```

## Troubleshooting

### Database connection issues
1. Ensure the database is healthy:
   ```bash
   docker-compose ps
   ```
2. Check logs:
   ```bash
   docker-compose logs db
   ```

### Backend startup issues
1. Check backend logs:
   ```bash
   docker-compose logs backend
   ```
2. Ensure `.env` file exists with correct values
3. Verify API_KEY is set for Google Gemini

### Reset everything
```bash
docker-compose down -v
rm -rf data/pgdata
docker-compose up --build -d
```

## Production Considerations
- Use secure passwords
- Consider using Docker secrets for sensitive data
- Set up proper backup for PostgreSQL data
- Configure reverse proxy (nginx/traefik) for HTTPS
- Adjust resource limits in docker-compose.yml
