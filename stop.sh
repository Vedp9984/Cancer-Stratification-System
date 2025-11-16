#!/bin/bash
# Stop all services and clean up

echo "Stopping Cancer Stratification System..."

# Stop services
docker compose down

echo "Services stopped!"
echo ""
echo "To remove all data (volumes), run:"
echo "  docker compose down -v"
echo ""
echo "To remove Docker images, run:"
echo "  docker compose down --rmi all"
