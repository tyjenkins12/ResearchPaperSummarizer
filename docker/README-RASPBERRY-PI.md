# Raspberry Pi Deployment Guide

Deploy the Research Paper Summarizer on Raspberry Pi using Portainer with optimized resource allocation.

## Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS 64-bit
- Docker and Docker Compose installed
- Portainer installed and running

## Quick Setup

### 1. Install Docker on Raspberry Pi

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt update
sudo apt install docker-compose-plugin

# Reboot to apply changes
sudo reboot
```

### 2. Install Portainer

```bash
# Create Portainer volume
sudo docker volume create portainer_data

# Run Portainer
sudo docker run -d -p 8080:8000 -p 9443:9443 \
    --name portainer --restart=always \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v portainer_data:/data \
    portainer/portainer-ce:latest
```

Access Portainer at: `https://your-pi-ip:9443`

### 3. Deploy via Portainer

1. **Build the Application Image:**
   ```bash
   # On your development machine or Pi
   git clone <your-repo>
   cd research-paper-summarizer
   git checkout raspberry-pi
   
   # Build ARM64 image
   docker build -f docker/Dockerfile -t papersum:raspberry-pi .
   ```

2. **Import Stack in Portainer:**
   - Go to Portainer web interface
   - Navigate to "Stacks"
   - Click "Add stack"
   - Name: `papersum-rpi`
   - Upload `docker/docker-compose.portainer.yml`
   - Set environment variables (see below)
   - Deploy

### 4. Environment Configuration

Create these environment variables in Portainer:

```env
# Email Settings (Required for newsletters)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
FROM_NAME=Research Paper Digest

# Optional: Database (defaults to SQLite)
# DATABASE_URL=postgresql://user:pass@postgres:5432/papersum
```

## Resource Allocation

### Memory Distribution (4GB Pi)
- **Total Available**: ~3.5GB (system reserves ~500MB)
- **Papersum App**: 1GB limit (512MB reserved)
- **Grobid**: 1GB limit (512MB reserved) 
- **System**: 1.5GB remaining

### CPU Allocation
- **Papersum**: 2 cores max, 0.5 cores reserved
- **Grobid**: 1 core max, 0.25 cores reserved

## Performance Optimizations

### 1. Model Selection
The ARM64 build automatically uses lightweight models:
- T5-small (240MB) instead of BART-large
- Sentence-BERT with reduced dimensions
- Single worker process

### 2. Processing Limits
- Max concurrent requests: 5
- Request timeout: 5 minutes
- Worker connections: 50

### 3. Grobid Optimization
- Reduced heap size: 1GB max
- Headless Java mode
- Extended health check intervals

## Monitoring

### Container Health
```bash
# Check container status
docker ps

# View logs
docker logs papersum-rpi
docker logs grobid-rpi

# Monitor resource usage
docker stats
```

### Application Health
- Main app: `http://your-pi-ip:8000`
- API docs: `http://your-pi-ip:8000/docs`
- Admin dashboard: `http://your-pi-ip:8000/admin`
- Grobid (internal): `http://grobid:8070`

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   ```bash
   # Check available memory
   free -h
   
   # Reduce Grobid memory if needed
   # Edit JAVA_OPTS=-Xmx512m in stack
   ```

2. **Slow Performance**
   ```bash
   # Enable swap if not already enabled
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   ```

3. **Grobid Fails to Start**
   ```bash
   # Check Grobid logs
   docker logs grobid-rpi
   
   # Increase startup timeout in health check
   ```

### Performance Tips

1. **Use SD Card Class 10+ or SSD**
2. **Enable GPU memory split:**
   ```bash
   # In /boot/config.txt
   gpu_mem=16  # Minimal for headless
   ```

3. **Monitor temperature:**
   ```bash
   vcgencmd measure_temp
   # Add cooling if > 70°C under load
   ```

## Scaling Options

### For Pi with 8GB RAM:
- Increase papersum memory to 2GB
- Increase Grobid to 1.5GB
- Enable multiple workers: `--workers 2`

### Cluster Setup:
- Use Docker Swarm with multiple Pi devices
- Distribute Grobid and app services
- Share storage via NFS

## Backup & Maintenance

### Data Backup
```bash
# Backup application data
docker run --rm -v papersum_data:/data -v $(pwd):/backup \
    alpine tar czf /backup/papersum-backup.tar.gz /data

# Backup model cache (optional)
docker run --rm -v model_cache:/cache -v $(pwd):/backup \
    alpine tar czf /backup/models-backup.tar.gz /cache
```

### Updates
```bash
# Update images
docker pull papersum:raspberry-pi
docker pull lfoppiano/grobid:0.8.0

# Restart stack in Portainer
```

## Security Notes

- Change default Portainer admin password
- Use environment files for secrets
- Consider VPN access for remote management
- Regular security updates: `sudo apt update && sudo apt upgrade`