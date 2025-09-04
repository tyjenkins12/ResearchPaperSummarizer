# Research Paper Summarizer - Deployment Guide

This guide covers deploying the Research Paper Summarizer application using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose v2.0+
- At least 4GB RAM available for containers
- 10GB+ disk space for models and data

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repository-url>
   cd research-paper-summarizer
   cp .env.example .env
   ```

2. **Configure environment:**
   Edit `.env` file with your email settings for newsletter functionality.

3. **Deploy:**
   ```bash
   # Development deployment
   docker/deploy.sh development
   
   # Production deployment  
   docker/deploy.sh production
   ```

4. **Access the application:**
   - Main app: http://localhost:8000
   - API docs: http://localhost:8000/docs
   - Admin dashboard: http://localhost:8000/admin

## Deployment Environments

### Development Environment

**Features:**
- Hot code reloading
- Exposed Grobid port for testing
- Reduced memory allocation
- SQLite database
- Debug logging enabled

**Start:**
```bash
docker/deploy.sh development
```

**Configuration:**
- Uses `docker-compose.dev.yml`
- Mounts source code as volumes
- Grobid accessible at localhost:8070

### Production Environment

**Features:**
- Optimized container images
- Multi-worker Uvicorn server
- Health checks
- Auto-restart policies
- Optional PostgreSQL and Redis

**Start:**
```bash
docker/deploy.sh production
```

**Configuration:**
- Uses `docker-compose.yml`
- Can enable PostgreSQL with `--profile postgres`
- Can enable Redis with `--profile redis`
- Can enable Nginx with `--profile nginx`

## Service Architecture

### Core Services

1. **papersum-app**: Main FastAPI application
2. **grobid-service**: PDF processing service

### Optional Services (Production)

3. **postgres**: PostgreSQL database (profile: postgres)
4. **redis**: Redis cache and job queue (profile: redis)  
5. **nginx**: Reverse proxy and SSL termination (profile: nginx)

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
FROM_NAME=Research Paper Digest

# Database (Production only)
POSTGRES_PASSWORD=secure_password_here

# AI Models
USE_LOCAL_MODELS=true
DEVICE=cpu  # or 'cuda' for GPU, 'mps' for Apple Silicon
```

### Email Setup

For Gmail:
1. Enable 2-factor authentication
2. Generate an app password
3. Use the app password as `SMTP_PASSWORD`

## Advanced Deployment Options

### With PostgreSQL Database

```bash
# Start with PostgreSQL
docker compose --profile postgres up -d

# Update connection string
export DATABASE_URL="postgresql://papersum:password@postgres:5432/papersum"
```

### With Redis Cache

```bash
# Start with Redis
docker compose --profile redis up -d
```

### With Nginx Reverse Proxy

```bash
# Configure nginx.conf first
docker compose --profile nginx up -d
```

### SSL/HTTPS Setup

1. Place SSL certificates in `./ssl/` directory
2. Update `nginx.conf` with certificate paths
3. Enable nginx profile

## Management Commands

### Service Control

```bash
# Check status
docker/deploy.sh --status

# View logs
docker/deploy.sh --logs

# Stop all services
docker/deploy.sh --stop

# Restart services
docker compose restart
```

### Database Management

```bash
# Backup SQLite database
docker cp papersum-app:/app/data/research_papers.db ./backup.db

# Restore database
docker cp ./backup.db papersum-app:/app/data/research_papers.db
```

### Model Cache Management

```bash
# View model cache
docker volume inspect research-paper-summarizer_model_cache

# Clear model cache (will re-download on restart)
docker volume rm research-paper-summarizer_model_cache
```

## Monitoring

### Health Checks

- Application: `curl http://localhost:8000/health`
- Grobid: `curl http://localhost:8070/api/isalive`

### Service Status

```bash
# Container status
docker compose ps

# Resource usage
docker stats

# Service logs
docker compose logs -f papersum
docker compose logs -f grobid
```

### Admin Dashboard

Access the admin dashboard at http://localhost:8000/admin for:
- System statistics
- User management
- Newsletter scheduler control
- Service health monitoring

## Troubleshooting

### Common Issues

**Grobid fails to start:**
```bash
# Check memory allocation
docker stats grobid-service

# Increase memory in docker-compose.yml:
environment:
  - JAVA_OPTS=-Xmx6g  # Increase from 4g
```

**PDF processing fails:**
```bash
# Test Grobid directly
curl -X POST -F "input=@test.pdf" http://localhost:8070/api/processFulltextDocument

# Check Grobid logs
docker compose logs grobid
```

**Email sending fails:**
```bash
# Test email configuration
python -c "
from src.papersum.newsletter.email_service import EmailService
service = EmailService()
result = service.send_test_email('test@example.com')
print('Success!' if result else 'Failed!')
"
```

### Log Analysis

```bash
# Application logs
docker compose logs papersum | grep ERROR

# Grobid logs  
docker compose logs grobid | grep -i error

# System resource usage
docker system df
docker system prune  # Clean up unused resources
```

### Performance Tuning

**For high-volume processing:**

1. **Increase worker processes:**
   ```yaml
   command: ["uvicorn", "papersum.web.main:app", "--workers", "8"]
   ```

2. **Allocate more memory to Grobid:**
   ```yaml
   environment:
     - JAVA_OPTS=-Xmx8g
   ```

3. **Enable Redis for caching:**
   ```bash
   docker compose --profile redis up -d
   ```

## Security Considerations

### Production Security

1. **Change default passwords:**
   ```bash
   # Generate secure passwords
   openssl rand -base64 32
   ```

2. **Restrict network access:**
   ```yaml
   # In docker-compose.yml, remove port mappings for internal services
   grobid:
     # ports:
     #   - "8070:8070"  # Remove this line
   ```

3. **Use secrets for sensitive data:**
   ```yaml
   # Use Docker secrets instead of environment variables
   secrets:
     db_password:
       file: ./secrets/db_password.txt
   ```

4. **Enable SSL/TLS:**
   ```bash
   # Use nginx profile with SSL certificates
   docker compose --profile nginx up -d
   ```

### Data Protection

- Regular database backups
- Encrypted storage volumes
- Network isolation between services
- Regular security updates

## Updates and Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker/deploy.sh production
```

### Update Dependencies

```bash
# Update Python packages
docker compose build --no-cache

# Update base images
docker compose pull
```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker cp papersum-app:/app/data ./backup_$DATE/
docker cp papersum-app:/root/.cache/papersum/models ./models_backup_$DATE/
```

## Support

For deployment issues:

1. Check logs: `docker/deploy.sh --logs`
2. Verify health: `curl http://localhost:8000/health`
3. Review configuration in `.env` file
4. Check system resources: `docker stats`

For application-specific issues:
- API documentation: http://localhost:8000/docs
- Admin dashboard: http://localhost:8000/admin