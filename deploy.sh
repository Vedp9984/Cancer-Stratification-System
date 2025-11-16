#!/bin/bash
# Quick deployment script for Cancer Stratification System

set -e

echo "======================================"
echo "Cancer Stratification System Deployment"
echo "======================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed!${NC}"
    echo "Please install Docker first:"
    echo "  sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running!${NC}"
    echo "Starting Docker..."
    sudo systemctl start docker
fi

# Check if user is in docker group
if ! groups | grep -q docker; then
    echo -e "${YELLOW}Warning: Your user is not in the docker group${NC}"
    echo "You may need to run commands with sudo or add yourself to the group:"
    echo "  sudo usermod -aG docker $USER"
    echo "  Then log out and log back in"
    echo ""
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓ .env file created${NC}"
    echo ""
    echo -e "${YELLOW}Please review .env and update if needed:${NC}"
    echo "  nano .env"
    echo ""
    read -p "Press Enter to continue..."
fi

# Build Docker images
echo -e "${GREEN}Building Docker images...${NC}"
docker compose build

# Start services
echo ""
echo -e "${GREEN}Starting services...${NC}"
docker compose up -d

# Wait for services to be healthy
echo ""
echo -e "${GREEN}Waiting for services to start...${NC}"
sleep 10

# Check service status
echo ""
echo -e "${GREEN}Service Status:${NC}"
docker compose ps

echo ""
echo -e "${GREEN}======================================"
echo "Deployment Complete!"
echo "======================================${NC}"
echo ""
echo "Access the application:"
echo "  Frontend: http://localhost"
echo "  Backend API: http://localhost:5000/api"
echo ""
echo "View logs:"
echo "  docker compose logs -f"
echo ""
echo "Stop services:"
echo "  docker compose down"
echo ""
echo -e "${YELLOW}Test Credentials:${NC}"
echo "  Patient: patient@test.com / password123"
echo "  Doctor: doctor@test.com / password123"
echo "  Radiologist: radiologist@test.com / password123"
echo ""
echo "For more information, see DEPLOYMENT_GUIDE.md"
