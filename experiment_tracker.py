#!/usr/bin/env python3
"""
Simple experiment tracker using Cloud Storage for coordination
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from gcp_config import GCPConfig, ExperimentJob, ExperimentStatus, get_gcp_config
from gcp_storage import GCPStorageManager

class ExperimentTracker:
    """Simple experiment tracker using Cloud Storage for persistence"""
    
    def __init__(self, config: Optional[GCPConfig] = None):
        self.config = config or get_gcp_config()
        self.storage = GCPStorageManager(config)
        self.jobs_file = "metadata/experiment_jobs.json"
        
    def initialize_experiments(self, datasets: List[Dict], models: List[str]) -> List[ExperimentJob]:
        """Initialize all experiment combinations"""
        jobs = []
        
        for dataset in datasets:
            for model_name in models:
                job_id = f"{model_name}_{dataset['name']}_{uuid.uuid4().hex[:8]}"
                job = ExperimentJob(
                    job_id=job_id,
                    model_name=model_name,
                    dataset_name=dataset['name'],
                    status=ExperimentStatus.PENDING
                )
                jobs.append(job)
        
        # Save initial job state
        self._save_jobs(jobs)
        print(f"✅ Initialized {len(jobs)} experiment jobs")
        return jobs
    
    def get_pending_jobs(self) -> List[ExperimentJob]:
        """Get all pending jobs"""
        jobs = self._load_jobs()
        return [job for job in jobs if job.status == ExperimentStatus.PENDING]
    
    def get_job_status(self, job_id: str) -> Optional[ExperimentJob]:
        """Get status of specific job"""
        jobs = self._load_jobs()
        for job in jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def update_job_status(self, job_id: str, status: ExperimentStatus, 
                         cloud_run_job_name: Optional[str] = None,
                         error_message: Optional[str] = None,
                         results_path: Optional[str] = None) -> None:
        """Update job status"""
        jobs = self._load_jobs()
        
        for job in jobs:
            if job.job_id == job_id:
                job.status = status
                
                if status == ExperimentStatus.RUNNING and not job.start_time:
                    job.start_time = datetime.now().isoformat()
                
                if status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]:
                    job.end_time = datetime.now().isoformat()
                
                if cloud_run_job_name:
                    job.cloud_run_job_name = cloud_run_job_name
                
                if error_message:
                    job.error_message = error_message
                    
                if results_path:
                    job.results_path = results_path
                
                break
        
        self._save_jobs(jobs)
    
    def get_experiment_summary(self) -> Dict:
        """Get summary of all experiments"""
        jobs = self._load_jobs()
        
        summary = {
            'total_jobs': len(jobs),
            'pending': len([j for j in jobs if j.status == ExperimentStatus.PENDING]),
            'running': len([j for j in jobs if j.status == ExperimentStatus.RUNNING]),
            'completed': len([j for j in jobs if j.status == ExperimentStatus.COMPLETED]),
            'failed': len([j for j in jobs if j.status == ExperimentStatus.FAILED]),
            'cancelled': len([j for j in jobs if j.status == ExperimentStatus.CANCELLED]),
            'last_updated': datetime.now().isoformat()
        }
        
        return summary
    
    def cleanup_stale_jobs(self, max_age_hours: int = 2) -> None:
        """Mark old running jobs as failed (for spot instance recovery)"""
        jobs = self._load_jobs()
        current_time = datetime.now()
        
        for job in jobs:
            if job.status == ExperimentStatus.RUNNING and job.start_time:
                start_time = datetime.fromisoformat(job.start_time.replace('Z', '+00:00'))
                age_hours = (current_time - start_time.replace(tzinfo=None)).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    print(f"🔄 Marking stale job as failed: {job.job_id} (age: {age_hours:.1f}h)")
                    job.status = ExperimentStatus.FAILED
                    job.error_message = f"Job timed out after {age_hours:.1f} hours (likely spot instance preemption)"
                    job.end_time = current_time.isoformat()
        
        self._save_jobs(jobs)
    
    def reset_failed_jobs(self) -> int:
        """Reset failed jobs to pending (for retry)"""
        jobs = self._load_jobs()
        reset_count = 0
        
        for job in jobs:
            if job.status == ExperimentStatus.FAILED:
                job.status = ExperimentStatus.PENDING
                job.error_message = None
                job.start_time = None
                job.end_time = None
                job.cloud_run_job_name = None
                reset_count += 1
        
        self._save_jobs(jobs)
        print(f"🔄 Reset {reset_count} failed jobs to pending")
        return reset_count
    
    def _load_jobs(self) -> List[ExperimentJob]:
        """Load jobs from Cloud Storage"""
        try:
            if self.storage.file_exists(self.jobs_file):
                data = self.storage.download_json(self.jobs_file)
                return [ExperimentJob.from_dict(job_data) for job_data in data.get('jobs', [])]
            else:
                return []
        except Exception as e:
            print(f"⚠️ Error loading jobs: {e}")
            return []
    
    def _save_jobs(self, jobs: List[ExperimentJob]) -> None:
        """Save jobs to Cloud Storage"""
        data = {
            'jobs': [job.to_dict() for job in jobs],
            'last_updated': datetime.now().isoformat()
        }
        self.storage.upload_json(data, self.jobs_file)
    
    def export_results_summary(self) -> Dict:
        """Export completed results summary"""
        jobs = self._load_jobs()
        completed_jobs = [job for job in jobs if job.status == ExperimentStatus.COMPLETED]
        
        results = []
        for job in completed_jobs:
            if job.results_path:
                try:
                    result_data = self.storage.download_json(job.results_path)
                    results.append(result_data)
                except Exception as e:
                    print(f"⚠️ Could not load results for {job.job_id}: {e}")
        
        summary = {
            'experiment_summary': self.get_experiment_summary(),
            'completed_results': results,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Save summary to Cloud Storage
        self.storage.upload_json(summary, 'results/experiment_summary.json')
        
        return summary 