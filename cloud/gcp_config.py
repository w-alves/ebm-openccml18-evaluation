#!/usr/bin/env python3
"""
GCP Configuration and Experiment Tracking
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum

class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class GCPConfig:
    """GCP configuration settings"""
    project_id: str
    region: str = "us-central1"
    bucket_name: str = ""
    
    # Cloud Run settings
    cloud_run_service_account: Optional[str] = None
    cloud_run_cpu: str = "8"
    cloud_run_memory: str = "16Gi"
    cloud_run_timeout: str = "30000s"  # 2 hour max per experiment
    
    # Container settings
    container_registry: str = ""
    image_name_py311: str = "ml-experiments-py311"
    image_name_py38: str = "ml-experiments-py38"
    image_tag: str = "latest"
    
    def __post_init__(self):
        if not self.bucket_name:
            self.bucket_name = f"{self.project_id}-ml-experiments"
        if not self.container_registry:
            self.container_registry = f"gcr.io/{self.project_id}"
    
    def get_image_for_model(self, model_name: str) -> str:
        """Get the appropriate container image for a given model"""
        if model_name == "AutoSklearn":
            return f"{self.container_registry}/{self.image_name_py38}:{self.image_tag}"
        else:
            return f"{self.container_registry}/{self.image_name_py311}:{self.image_tag}"

def get_gcp_config() -> GCPConfig:
    """Get GCP configuration from environment variables"""
    project_id = "mestrado-461619"

    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is required")
    
    return GCPConfig(
        project_id=project_id,
        region=os.getenv('GCP_REGION', 'us-central1'),
        bucket_name=os.getenv('GCP_BUCKET', f"{project_id}-ml-experiments"),
        cloud_run_service_account=os.getenv('GCP_SERVICE_ACCOUNT'),
        cloud_run_cpu=os.getenv('CLOUD_RUN_CPU', '8'),
        cloud_run_memory=os.getenv('CLOUD_RUN_MEMORY', '16Gi'),
        cloud_run_timeout=os.getenv('CLOUD_RUN_TIMEOUT', '30000s'),
    )

@dataclass
class ExperimentJob:
    """Represents a single model+dataset experiment job"""
    job_id: str
    model_name: str
    dataset_name: str
    status: ExperimentStatus
    cloud_run_job_name: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    results_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'status': self.status.value,
            'cloud_run_job_name': self.cloud_run_job_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'error_message': self.error_message,
            'results_path': self.results_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentJob':
        return cls(
            job_id=data['job_id'],
            model_name=data['model_name'],
            dataset_name=data['dataset_name'],
            status=ExperimentStatus(data['status']),
            cloud_run_job_name=data.get('cloud_run_job_name'),
            start_time=data.get('start_time'),
            end_time=data.get('end_time'),
            error_message=data.get('error_message'),
            results_path=data.get('results_path')
        ) 