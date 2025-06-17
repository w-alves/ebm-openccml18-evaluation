#!/bin/bash

# GCP ML Experiments Deployment Script
set -e

echo "🚀 GCP ML Experiments Deployment"
echo "================================="

# Check required environment variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "❌ Please set GCP_PROJECT_ID environment variable"
    exit 1
fi

# Set default values
GCP_REGION=${GCP_REGION:-"us-central1"}
IMAGE_NAME_PY311=${IMAGE_NAME_PY311:-"ml-experiments-py311"}
IMAGE_NAME_PY38=${IMAGE_NAME_PY38:-"ml-experiments-py38"}
IMAGE_TAG=${IMAGE_TAG:-"v4"}

echo "📋 Configuration:"
echo "  Project ID: $GCP_PROJECT_ID"
echo "  Region: $GCP_REGION"
echo "  Python 3.11 Image: $IMAGE_NAME_PY311:$IMAGE_TAG"
echo "  Python 3.8 Image: $IMAGE_NAME_PY38:$IMAGE_TAG"
echo ""

# Enable required APIs
echo "🔧 Enabling required GCP APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$GCP_PROJECT_ID
gcloud services enable run.googleapis.com --project=$GCP_PROJECT_ID
gcloud services enable storage.googleapis.com --project=$GCP_PROJECT_ID

echo "✅ APIs enabled"

# Configure Docker for GCP
echo "🐳 Configuring Docker for GCP..."
gcloud auth configure-docker --quiet

# Build and push Docker images
echo "📦 Building Docker images with no cache..."
echo "  Building Python 3.11 image..."
docker build -f Dockerfile -t gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY311:$IMAGE_TAG .

echo "  Building Python 3.8 image..."
docker build -f Dockerfile.py38 -t gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY38:$IMAGE_TAG .

echo "📤 Pushing Docker images to GCR..."
docker push gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY311:$IMAGE_TAG
docker push gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY38:$IMAGE_TAG

echo "✅ Docker images deployed to gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY311:$IMAGE_TAG and gcr.io/$GCP_PROJECT_ID/$IMAGE_NAME_PY38:$IMAGE_TAG"

# Create service account (if it doesn't exist)
SERVICE_ACCOUNT_NAME="ml-experiments-runner"
SERVICE_ACCOUNT_EMAIL="$SERVICE_ACCOUNT_NAME@$GCP_PROJECT_ID.iam.gserviceaccount.com"

echo "🔐 Setting up service account..."
if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL --project=$GCP_PROJECT_ID >/dev/null 2>&1; then
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="ML Experiments Runner" \
        --project=$GCP_PROJECT_ID
    echo "✅ Created service account: $SERVICE_ACCOUNT_EMAIL"
else
    echo "✅ Service account already exists: $SERVICE_ACCOUNT_EMAIL"
fi

# Grant necessary permissions
echo "🔑 Granting permissions..."
gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $GCP_PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/iam.serviceAccountUser"

echo "✅ Permissions granted"

# Export environment variables for the launcher
export GCP_SERVICE_ACCOUNT=$SERVICE_ACCOUNT_EMAIL

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "To run the experiments:"
echo "  1. Make sure your datasets are prepared locally (run preprocess.py, split_data.py)"
echo "  2. Run: python cloud_job_launcher.py"
echo ""
echo "To monitor experiments:"
echo "  - Check Cloud Run jobs in the GCP Console"
echo "  - Monitor logs with: gcloud logging read 'resource.type=cloud_run_job' --project=$GCP_PROJECT_ID"
echo "" 