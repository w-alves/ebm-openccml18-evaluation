# ML Experiments on GCP

This project has been migrated to run on Google Cloud Platform (GCP) with full parallelization of model+dataset combinations using Cloud Run jobs. Each experiment runs independently and can handle spot instance interruptions gracefully.

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Local Machine     │    │   Cloud Storage      │    │   Cloud Run Jobs    │
│                     │    │                      │    │                     │
│ • cloud_job_launcher│───▶│ • Datasets           │◄───│ • Individual        │
│ • Uploads data      │    │ • Model code         │    │   Experiments       │
│ • Submits jobs      │    │ • Results            │    │ • Fault tolerant    │
│ • Monitors progress │    │ • Job coordination   │    │ • Auto-retry        │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Key Features

- **Parallel Execution**: Each model+dataset combination runs as a separate Cloud Run job
- **Multi-Python Support**: Automatic Python 3.8 for Auto-sklearn, Python 3.11 for modern models
- **Fault Tolerance**: Jobs automatically retry on failure (e.g., spot instance preemption)
- **Cost Effective**: Uses Cloud Run's serverless billing (pay only for execution time)
- **Simple Architecture**: Minimal infrastructure, perfect for POC/research
- **State Management**: Persistent tracking via Cloud Storage

## Prerequisites

1. **GCP Project** with billing enabled
2. **Google Cloud SDK** installed and authenticated
3. **Docker** installed locally
4. **Python 3.11+** with dependencies

### GCP Setup

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud auth application-default login

# Set your project
export GCP_PROJECT_ID="your-project-id"
gcloud config set project $GCP_PROJECT_ID
```

## Quick Start

### 1. Prepare Data Locally

First, run the data preparation pipeline locally:

```bash
# Fetch datasets
python fetch_datasets.py

# Preprocess datasets  
python preprocess.py

# Split into train/test
python split_data.py
```

### 2. Deploy to GCP

```bash
# Set project ID
export GCP_PROJECT_ID="your-project-id"

# Deploy infrastructure and Docker image
chmod +x deploy_gcp.sh
./deploy_gcp.sh
```

This script will:
- Enable required GCP APIs
- Build and push Docker images to Container Registry (Python 3.8 & 3.11)
- Create service accounts with proper permissions
- Set up Cloud Storage bucket

### 3. Launch Experiments

```bash
# Run the experiment launcher
python cloud_job_launcher.py
```

This will:
- Upload datasets and code to Cloud Storage
- Create experiment tracking jobs (3 datasets × 6 models = 18 jobs)
- Submit Cloud Run jobs in parallel batches
- Monitor progress and handle failures

## Configuration

### Environment Variables

```bash
# Required
export GCP_PROJECT_ID="your-project-id"

# Optional
export GCP_REGION="us-central1"           # Default region
export GCP_BUCKET="custom-bucket-name"    # Custom bucket name
export CLOUD_RUN_CPU="4"                  # CPU allocation per job
export CLOUD_RUN_MEMORY="16Gi"            # Memory allocation per job
export CLOUD_RUN_TIMEOUT="3600s"          # Max execution time per job
```

### Experiment Configuration

The system automatically creates all combinations of:

- **Models**: AutoSklearn (Python 3.8), AutoGluon, XGBoost, LightGBM, CatBoost, EBM (Python 3.11)
- **Datasets**: All datasets in `data/split/`

Each combination runs as an independent Cloud Run job with the appropriate Python environment.

## Monitoring

### 1. Local Monitoring

The launcher provides real-time status updates:

```bash
📊 Status: 2 pending, 3 running, 8 completed, 2 failed
```

### 2. GCP Console

- **Cloud Run**: https://console.cloud.google.com/run
- **Cloud Storage**: https://console.cloud.google.com/storage
- **Logs**: https://console.cloud.google.com/logs

### 3. Command Line

```bash
# View Cloud Run jobs
gcloud run jobs list --region=$GCP_REGION

# View logs for specific job
gcloud logs read 'resource.type=cloud_run_job resource.labels.job_name="ml-exp-xgboost-dataset-1"' \
    --project=$GCP_PROJECT_ID

# Monitor all experiment logs
gcloud logs read 'resource.type=cloud_run_job' --project=$GCP_PROJECT_ID
```

## Results

Results are automatically saved to Cloud Storage:

```
gs://your-bucket/
├── datasets/                 # Input datasets
├── code/                     # Model implementations  
├── metadata/
│   ├── experiment_config.json    # Experiment configuration
│   └── experiment_jobs.json      # Job tracking
└── results/
    ├── raw/                       # Individual results
    │   ├── XGBoost_kc2.json
    │   ├── AutoGluon_kc2.json
    │   └── ...
    └── experiment_summary.json    # Aggregated results
```

### Download Results

```bash
# Download all results
gsutil -m cp -r gs://your-bucket/results/ ./results/

# Download summary only
gsutil cp gs://your-bucket/results/experiment_summary.json .
```

## Cost Optimization

This architecture is designed for cost efficiency:

1. **Serverless**: Only pay for actual execution time
2. **Parallel**: Faster completion = lower total cost
3. **Fault Tolerance**: No wasted compute on failed jobs
4. **Resource Optimization**: Right-sized containers per job

### Estimated Costs

For typical workloads (assuming 1-hour experiments):
- **Cloud Run**: ~$0.10-0.20 per experiment
- **Cloud Storage**: ~$0.01-0.05 total
- **Total**: ~$2.40-4.80 for full experiment suite (3 datasets × 6 models = 18 jobs)

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   ```bash
   # Make sure Docker is running
   docker --version
   
   # Check disk space
   df -h
   ```

2. **Permission Denied**
   ```bash
   # Re-authenticate
   gcloud auth application-default login
   
   # Check project access
   gcloud projects describe $GCP_PROJECT_ID
   ```

3. **Jobs Fail to Start**
   ```bash
   # Check service account permissions
   gcloud projects get-iam-policy $GCP_PROJECT_ID
   
   # Verify Cloud Run is enabled
   gcloud services list --enabled | grep run
   ```

4. **Out of Memory**
   ```bash
   # Increase memory allocation
   export CLOUD_RUN_MEMORY="32Gi"
   
   # Or reduce dataset size in preprocessing
   ```

### Failed Job Recovery

The system automatically handles failed jobs:

```python
# Reset failed jobs to pending (for retry)
from experiment_tracker import ExperimentTracker
tracker = ExperimentTracker()
tracker.reset_failed_jobs()
```

### Manual Job Management

```bash
# List all jobs
gcloud run jobs list --region=$GCP_REGION

# Delete a specific job
gcloud run jobs delete JOB_NAME --region=$GCP_REGION

# Execute a job manually
gcloud run jobs execute JOB_NAME --region=$GCP_REGION
```

## Development

### Local Testing

Test the experiment runner locally:

```bash
# Set environment variables
export JOB_ID="test-job"
export MODEL_NAME="XGBoost" 
export DATASET_NAME="dataset_1063_kc2"
export GCP_PROJECT_ID="your-project-id"

# Run locally
python cloud_experiment_runner.py
```

### Custom Models

To add new models:

1. Create model class in `models/` following the base interface
2. Add to model list in `cloud_job_launcher.py`
3. Update Docker image and redeploy

### Infrastructure as Code

For production use, consider using Terraform:

```hcl
resource "google_cloud_run_v2_job" "ml_experiment" {
  name     = "ml-experiment-template"
  location = var.region
  
  template {
    template {
      containers {
        image = var.container_image
        resources {
          limits = {
            cpu    = "4"
            memory = "16Gi"
          }
        }
      }
    }
  }
}
```

## Migration from Local

This replaces the original multi-environment setup:

| Original | GCP Version |
|----------|-------------|
| `conda` environments | Docker containers |
| `launch_experiments.sh` | `cloud_job_launcher.py` |
| Local file system | Cloud Storage |
| Sequential execution | Parallel Cloud Run jobs |
| Manual coordination | Automatic tracking |

The core ML models and evaluation logic remain unchanged.

## Support

For issues:
1. Check the troubleshooting section above
2. Review Cloud Run and Cloud Storage logs
3. Verify GCP quotas and billing
4. Test with a single job first before running full suite 