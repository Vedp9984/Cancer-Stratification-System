#!/bin/bash
# Backup script for Cancer Stratification System

BACKUP_DIR="/home/$(whoami)/cancer-app-backups"
DATE=$(date +%Y%m%d_%H%M%S)
MONGO_PASSWORD=${MONGO_ROOT_PASSWORD:-adminpassword123}

mkdir -p $BACKUP_DIR

echo "Starting backup..."

# Backup MongoDB
docker exec cancer-stratification-db mongodump \
  --username admin \
  --password $MONGO_PASSWORD \
  --authenticationDatabase admin \
  --db cancer_stratification \
  --archive=/tmp/backup.archive \
  --gzip

docker cp cancer-stratification-db:/tmp/backup.archive \
  $BACKUP_DIR/mongodb_backup_$DATE.archive.gz

# Backup uploaded images
docker run --rm \
  -v nnn_for_cancer_backend_uploads:/data \
  -v $BACKUP_DIR:/backup \
  ubuntu tar czf /backup/uploads_backup_$DATE.tar.gz /data

echo "Backup completed!"
echo "Location: $BACKUP_DIR"
echo "Files:"
echo "  - mongodb_backup_$DATE.archive.gz"
echo "  - uploads_backup_$DATE.tar.gz"

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -name "*.gz" -mtime +7 -delete
echo "Old backups cleaned up (kept last 7 days)"
