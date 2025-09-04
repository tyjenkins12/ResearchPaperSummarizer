#!/bin/bash

# Deployment script for Research Paper Summarizer
# Usage: ./deploy.sh [development|production]

set -e

# Configuration
ENVIRONMENT=${1:-development}
PROJECT_NAME="research-paper-summarizer"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null && ! docker-compose --version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for: $ENVIRONMENT"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_warning "Please configure .env file with your settings"
    fi
    
    # Create necessary directories
    mkdir -p data/uploads data/outputs logs
    
    log_success "Environment setup complete"
}

# Build and deploy
deploy() {
    log_info "Starting deployment..."
    
    case $ENVIRONMENT in
        development|dev)
            log_info "Deploying development environment..."
            docker compose -f docker-compose.dev.yml down --remove-orphans
            docker compose -f docker-compose.dev.yml build --no-cache
            docker compose -f docker-compose.dev.yml up -d
            ;;
        production|prod)
            log_info "Deploying production environment..."
            docker compose -f docker-compose.yml down --remove-orphans
            docker compose -f docker-compose.yml build --no-cache
            docker compose -f docker-compose.yml up -d
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            log_info "Usage: $0 [development|production]"
            exit 1
            ;;
    esac
}

# Wait for services
wait_for_services() {
    log_info "Waiting for services to be ready..."
    
    # Wait for Grobid
    log_info "Waiting for Grobid..."
    timeout=180
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8070/api/isalive &> /dev/null; then
            log_success "Grobid is ready"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    
    if [ $elapsed -ge $timeout ]; then
        log_error "Grobid failed to start within ${timeout}s"
        exit 1
    fi
    
    # Wait for main application
    log_info "Waiting for main application..."
    timeout=120
    elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "Application is ready"
            break
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo -n "."
    done
    
    if [ $elapsed -ge $timeout ]; then
        log_error "Application failed to start within ${timeout}s"
        exit 1
    fi
}

# Show status
show_status() {
    log_info "Service Status:"
    echo
    
    case $ENVIRONMENT in
        development|dev)
            docker compose -f docker-compose.dev.yml ps
            ;;
        production|prod)
            docker compose -f docker-compose.yml ps
            ;;
    esac
    
    echo
    log_info "Application URLs:"
    echo "  - Main application: http://localhost:8000"
    echo "  - API documentation: http://localhost:8000/docs"
    echo "  - Health check: http://localhost:8000/health"
    echo "  - Grobid service: http://localhost:8070"
    echo
}

# Test deployment
test_deployment() {
    log_info "Running deployment tests..."
    
    # Test main application
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Main application is responding"
    else
        log_error "Main application is not responding"
        return 1
    fi
    
    # Test Grobid
    if curl -f http://localhost:8070/api/isalive &> /dev/null; then
        log_success "Grobid service is responding"
    else
        log_error "Grobid service is not responding"
        return 1
    fi
    
    log_success "All deployment tests passed"
}

# Show help
show_help() {
    echo "Research Paper Summarizer Deployment Script"
    echo
    echo "Usage: $0 [ENVIRONMENT] [OPTIONS]"
    echo
    echo "Environments:"
    echo "  development, dev  Deploy for development (default)"
    echo "  production, prod  Deploy for production"
    echo
    echo "Options:"
    echo "  --help, -h       Show this help message"
    echo "  --status         Show service status only"
    echo "  --stop           Stop all services"
    echo "  --logs           Show service logs"
    echo
    echo "Examples:"
    echo "  $0 development    # Deploy development environment"
    echo "  $0 production     # Deploy production environment"
    echo "  $0 --status       # Show current status"
    echo "  $0 --stop         # Stop all services"
}

# Handle command line arguments
case ${1:-} in
    --help|-h)
        show_help
        exit 0
        ;;
    --status)
        show_status
        exit 0
        ;;
    --stop)
        log_info "Stopping all services..."
        docker compose -f docker-compose.dev.yml down 2>/dev/null || true
        docker compose -f docker-compose.yml down 2>/dev/null || true
        log_success "All services stopped"
        exit 0
        ;;
    --logs)
        log_info "Showing service logs..."
        if docker compose -f docker-compose.dev.yml ps -q &>/dev/null; then
            docker compose -f docker-compose.dev.yml logs -f
        else
            docker compose -f docker-compose.yml logs -f
        fi
        exit 0
        ;;
esac

# Main deployment flow
main() {
    echo "========================================="
    echo "Research Paper Summarizer Deployment"
    echo "Environment: $ENVIRONMENT"
    echo "========================================="
    
    check_prerequisites
    setup_environment
    deploy
    wait_for_services
    show_status
    test_deployment
    
    echo
    log_success "Deployment completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Configure your .env file with email settings"
    echo "2. Upload some PDF papers at http://localhost:8000/upload"
    echo "3. Check newsletter settings at http://localhost:8000/newsletter/settings"
    echo "4. Monitor the system at http://localhost:8000/admin"
}

# Run main function
main