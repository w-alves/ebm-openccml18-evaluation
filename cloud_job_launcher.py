#!/usr/bin/env python3
"""
Cloud Job Launcher - Submits individual experiments as Cloud Run jobs
"""

import json
import time
import asyncio
import subprocess
import tempfile
from typing import List, Dict
        
from gcp_config import get_gcp_config, ExperimentStatus
from gcp_storage import GCPStorageManager, prepare_datasets_for_cloud
from experiment_tracker import ExperimentTracker

class CloudJobLauncher:
    """Launches Cloud Run jobs for ML experiments"""
    
    def __init__(self, test_mode: bool = False):
        self.config = get_gcp_config()
        self.storage = GCPStorageManager(self.config)
        self.tracker = ExperimentTracker(self.config)
        self.test_mode = test_mode
        
    def setup_infrastructure(self, max_datasets: int = None, test_models: List[str] = None, force_upload: bool = False) -> None:
        """Setup GCP infrastructure (bucket, upload data)
        
        Args:
            max_datasets: Maximum number of datasets to use (None for all)
            test_models: List of models to use for testing (None for all)
            force_upload: Force re-upload of all files even if they exist
        """
        print("🔧 Setting up GCP infrastructure...")
        
        # Setup Cloud Storage bucket
        self.storage.setup_bucket()
        
        # Upload datasets and code
        all_datasets = prepare_datasets_for_cloud(self.storage, force_upload=force_upload)
        
        if not all_datasets:
            raise ValueError("No datasets available! Run data preparation first.")
        
        # Limit datasets if specified
        if max_datasets is not None:
            datasets = all_datasets[:max_datasets]
            print(f"🧪 Test mode: Using {len(datasets)} dataset(s) out of {len(all_datasets)} available")
        else:
            datasets = all_datasets
        
        # Set models based on test mode or provided list

        if self.test_mode:
            models = ['AutoGluon']  # Fast models for testing
            print(f"🧪 Test mode: Using models {models}")
        else:
            # [ 'XGBoost', 'LightGBM', 'CatBoost', 'EBM']
            models = ['AutoSklearn', 'AutoGluon']
            print(f"🚀 Full mode: Using all models {models}")
        
        # Initialize experiment tracking
        jobs = self.tracker.initialize_experiments(datasets, models)
        
        print(f"✅ Infrastructure setup complete. {len(jobs)} experiments ready to run.")
        if self.test_mode or max_datasets is not None or test_models is not None:
            print(f"📊 Test configuration: {len(datasets)} dataset(s) × {len(models)} models = {len(jobs)} experiments")
        
    def create_cloud_run_job(self, job_id: str, model_name: str, dataset_name: str) -> str:
        """Create a Cloud Run job for a single experiment using gcloud CLI"""
        
        print(f"  🏗️  create_cloud_run_job: Starting for {model_name} on {dataset_name}")
        
        # Sanitize job name to meet GCP requirements:
        # - Only lowercase alphanumeric characters and dashes
        # - Cannot begin or end with a dash
        # - Max 63 characters
        sanitized_name = job_id.lower()
        # Replace any non-alphanumeric characters with dashes
        sanitized_name = ''.join(c if c.isalnum() else '-' for c in sanitized_name)
        # Replace multiple consecutive dashes with a single dash
        while '--' in sanitized_name:
            sanitized_name = sanitized_name.replace('--', '-')
        # Remove leading/trailing dashes
        sanitized_name = sanitized_name.strip('-')
        # Ensure max length of 63 chars
        job_name = f"mlexp-{sanitized_name}"[:63].strip('-')
        
        print(f"  🏷️  create_cloud_run_job: Generated job name: {job_name}")
        
        # Container image - choose based on model
        image_uri = self.config.get_image_for_model(model_name)
        print(f"  🐳 create_cloud_run_job: Using image: {image_uri}")
        
        # Create a temporary YAML file for the job configuration
        job_config = {
            "apiVersion": "run.googleapis.com/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {
                    "experiment": "ml-automl",
                    "model": model_name.lower().replace(" ", "-"),
                    "dataset": dataset_name.lower().replace(" ", "-").replace("_", "-")
                }
            },
            "spec": {
                "template": {
                    "spec": {
                        "template": {
                            "spec": {
                                "restartPolicy": "Never",
                                "timeoutSeconds": 30000,
                                "serviceAccountName": self.config.cloud_run_service_account,
                                "containers": [
                                    {
                                        "image": image_uri,
                                        "command": ["python", "cloud_experiment_runner.py"],
                                        "env": [
                                            {"name": "JOB_ID", "value": job_id},
                                            {"name": "MODEL_NAME", "value": model_name},
                                            {"name": "DATASET_NAME", "value": dataset_name},
                                            {"name": "GCP_PROJECT_ID", "value": self.config.project_id},
                                            {"name": "GCP_BUCKET", "value": self.config.bucket_name},
                                            {"name": "GCP_REGION", "value": self.config.region},
                                        ],
                                        "resources": {
                                            "limits": {
                                                "cpu": self.config.cloud_run_cpu,
                                                "memory": self.config.cloud_run_memory
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            }
        }

        try:
            print(f"  📝 create_cloud_run_job: Creating temporary YAML config...")
            # Write job config to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                import yaml
                yaml.dump(job_config, f)
                config_file = f.name
            print(f"  ✅ create_cloud_run_job: YAML config created at {config_file}")

            # Create the job using gcloud CLI
            print(f"  🛠️  create_cloud_run_job: Building gcloud command...")
            cmd = [
                'gcloud', 'run', 'jobs', 'create', job_name,
                '--image', image_uri,
                '--task-timeout', self.config.cloud_run_timeout,
                '--max-retries', '0',
                '--parallelism', '0',
                '--cpu', self.config.cloud_run_cpu,
                '--memory', self.config.cloud_run_memory,
                '--region', self.config.region,
                '--project', self.config.project_id,
                '--set-env-vars', f'JOB_ID={job_id},MODEL_NAME={model_name},DATASET_NAME={dataset_name},GCP_PROJECT_ID={self.config.project_id},GCP_BUCKET={self.config.bucket_name},GCP_REGION={self.config.region}'
            ]
            
            # Add service account if specified
            if self.config.cloud_run_service_account:
                cmd.extend(['--service-account', self.config.cloud_run_service_account])
                print(f"  🔐 create_cloud_run_job: Added service account: {self.config.cloud_run_service_account}")
            
            # Add command to run
            cmd.extend(['--args', 'python,cloud_experiment_runner.py'])
            
            print(f"  🚀 create_cloud_run_job: Executing gcloud command...")
            print(f"  📋 create_cloud_run_job: Command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ create_cloud_run_job: Command executed successfully")
            print(f"  📤 create_cloud_run_job: stdout: {result.stdout[:200]}...")
            print(f"🚀 Created Cloud Run job: {job_name}")
            return job_name
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ create_cloud_run_job: gcloud command failed")
            print(f"❌ Failed to create job {job_name}: {e}")
            print(f"❌ stderr: {e.stderr}")
            print(f"❌ stdout: {e.stdout}")
            raise
        except ImportError:
            # Fallback: create job using gcloud without YAML
            print("  ⚠️  create_cloud_run_job: PyYAML not available, using simpler gcloud approach")
            print("⚠️ PyYAML not available, using simpler gcloud approach")
            return self._create_job_simple_gcloud(job_id, model_name, dataset_name, job_name, image_uri)
        except Exception as e:
            print(f"  ❌ create_cloud_run_job: Unexpected error: {e}")
            print(f"❌ Failed to create job {job_name}: {e}")
            raise
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(config_file)
                print(f"  🗑️  create_cloud_run_job: Cleaned up temp file {config_file}")
            except:
                pass

    def _create_job_simple_gcloud(self, job_id: str, model_name: str, dataset_name: str, job_name: str, image_uri: str) -> str:
        """Fallback method to create job using simple gcloud command"""
        
        print(f"  🔧 _create_job_simple_gcloud: Starting fallback method for {job_name}")
        
        try:
            # Create the job using gcloud CLI (simpler approach)
            cmd = [
                'gcloud', 'run', 'jobs', 'create', job_name,
                '--image', image_uri,
                '--task-timeout', self.config.cloud_run_timeout,
                '--max-retries', '0',
                '--parallelism', '1',
                '--cpu', self.config.cloud_run_cpu,
                '--memory', self.config.cloud_run_memory,
                '--region', self.config.region,
                '--project', self.config.project_id,
                '--set-env-vars', f'JOB_ID={job_id},MODEL_NAME={model_name},DATASET_NAME={dataset_name},GCP_PROJECT_ID={self.config.project_id},GCP_BUCKET={self.config.bucket_name},GCP_REGION={self.config.region}',
                '--args', 'python,cloud_experiment_runner.py'
            ]
            
            # Add service account if specified
            if self.config.cloud_run_service_account:
                cmd.extend(['--service-account', self.config.cloud_run_service_account])
            
            print(f"  🚀 _create_job_simple_gcloud: Executing fallback gcloud command...")
            print(f"  📋 _create_job_simple_gcloud: Command: {' '.join(cmd)}")
            
            # Execute the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ _create_job_simple_gcloud: Command executed successfully")
            print(f"🚀 Created Cloud Run job: {job_name}")
            return job_name
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ _create_job_simple_gcloud: gcloud command failed")
            print(f"❌ Failed to create job {job_name}: {e}")
            print(f"❌ stderr: {e.stderr}")
            print(f"❌ stdout: {e.stdout}")
            raise
    
    def execute_job(self, cloud_run_job_name: str) -> None:
        """Execute a Cloud Run job using gcloud CLI"""
        
        print(f"  ▶️  execute_job: Starting execution for {cloud_run_job_name}")
        
        try:
            # Remove --wait flag to make this non-blocking
            cmd = [
                'gcloud', 'run', 'jobs', 'execute', cloud_run_job_name,
                '--region', self.config.region,
                '--project', self.config.project_id
                # Removed --wait flag to prevent hanging
            ]
            
            print(f"  🚀 execute_job: Executing gcloud command (non-blocking)...")
            print(f"  📋 execute_job: Command: {' '.join(cmd)}")
            print(f"  💡 execute_job: Job will run asynchronously in the cloud")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"  ✅ execute_job: Command completed successfully")
            print(f"  📤 execute_job: stdout: {result.stdout[:200]}...")
            print(f"▶️ Executed job: {cloud_run_job_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ execute_job: gcloud command failed")
            print(f"❌ Failed to execute job {cloud_run_job_name}: {e}")
            print(f"❌ stderr: {e.stderr}")
            print(f"❌ stdout: {e.stdout}")
            raise
        except Exception as e:
            print(f"  ❌ execute_job: Unexpected error: {e}")
            print(f"❌ Failed to execute job {cloud_run_job_name}: {e}")
            raise
    
    async def launch_all_experiments(self, max_concurrent: int = 10, batch_delay: int = 30) -> None:
        """Launch all pending experiments with controlled concurrency"""
        print(f"🚀 Launching experiments with max {max_concurrent} concurrent jobs...")
        
        pending_jobs = self.tracker.get_pending_jobs()
        print(f"📋 Found {len(pending_jobs)} pending experiments")
        
        if not pending_jobs:
            print("✅ No pending experiments to run")
            return
        
        # Launch jobs in batches
        for i in range(0, len(pending_jobs), max_concurrent):
            batch = pending_jobs[i:i + max_concurrent]
            print(f"\n🔄 Processing batch {i//max_concurrent + 1}: {len(batch)} jobs")
            
            # Log the jobs in this batch
            for j, job in enumerate(batch):
                print(f"  📝 Batch job {j+1}: {job.model_name} on {job.dataset_name} (ID: {job.job_id})")
            
            print(f"  🎬 Starting batch execution...")
            batch_tasks = []
            for j, job in enumerate(batch):
                print(f"  🔧 Creating task {j+1} for {job.model_name} on {job.dataset_name}")
                task = self._launch_single_experiment(job, batch_index=i//max_concurrent + 1, job_index=j+1)
                batch_tasks.append(task)
            
            print(f"  ⏳ Waiting for {len(batch_tasks)} tasks to complete...")
            # Wait for batch to complete
            results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Log results
            success_count = 0
            error_count = 0
            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"  ❌ Task {j+1} failed: {result}")
                    error_count += 1
                else:
                    success_count += 1
            
            print(f"  ✅ Batch {i//max_concurrent + 1} complete: {success_count} succeeded, {error_count} failed")
            
            # Delay between batches to avoid overwhelming the system
            if i + max_concurrent < len(pending_jobs):
                print(f"⏳ Waiting {batch_delay}s before next batch...")
                await asyncio.sleep(batch_delay)
        
        print("🎉 All experiments launched!")
    
    async def _launch_single_experiment(self, job, batch_index: int = 0, job_index: int = 0) -> None:
        """Launch a single experiment job"""
        job_prefix = f"[B{batch_index}J{job_index}]"
        
        try:
            print(f"{job_prefix} 🔨 Starting job creation for {job.model_name} on {job.dataset_name}")
            
            # Create Cloud Run job
            print(f"{job_prefix} 📦 Creating Cloud Run job...")
            cloud_run_job_name = self.create_cloud_run_job(
                job.job_id, 
                job.model_name, 
                job.dataset_name
            )
            print(f"{job_prefix} ✅ Cloud Run job created: {cloud_run_job_name}")
            
            # Update tracker with Cloud Run job name
            print(f"{job_prefix} 📝 Updating job status in tracker...")
            self.tracker.update_job_status(
                job.job_id,
                ExperimentStatus.PENDING,
                cloud_run_job_name=cloud_run_job_name
            )
            print(f"{job_prefix} ✅ Job status updated")
            
            # Small delay before execution
            print(f"{job_prefix} ⏳ Waiting 2s before execution...")
            await asyncio.sleep(0.1)
            
            # Execute the job
            print(f"{job_prefix} ▶️ Executing Cloud Run job...")
            self.execute_job(cloud_run_job_name)
            print(f"{job_prefix} ✅ Job execution started")
            
            print(f"{job_prefix} 🎉 Successfully launched: {job.model_name} on {job.dataset_name}")
            
        except Exception as e:
            print(f"{job_prefix} ❌ Failed to launch {job.job_id}: {e}")
            print(f"{job_prefix} 📊 Exception type: {type(e).__name__}")
            import traceback
            print(f"{job_prefix} 🔍 Traceback:")
            traceback.print_exc()
            
            try:
                print(f"{job_prefix} 📝 Updating job status to FAILED...")
                self.tracker.update_job_status(
                    job.job_id,
                    ExperimentStatus.FAILED,
                    error_message=f"Failed to launch: {str(e)}"
                )
                print(f"{job_prefix} ✅ Job status updated to FAILED")
            except Exception as tracker_error:
                print(f"{job_prefix} ❌ Failed to update job status: {tracker_error}")
            
            # Re-raise the exception so it's captured in the gather results
            raise
    
    def monitor_experiments(self, check_interval: int = 300) -> None:
        """Monitor running experiments and handle failures"""
        print(f"👀 Starting experiment monitoring (check every {check_interval}s)...")
        
        while True:
            try:
                # Cleanup stale jobs (likely from spot instance preemption)
                self.tracker.cleanup_stale_jobs(max_age_hours=2)
                
                # Get summary
                summary = self.tracker.get_experiment_summary()
                print(f"📊 Status: {summary['pending']} pending, {summary['running']} running, "
                      f"{summary['completed']} completed, {summary['failed']} failed")
                
                # Check if all experiments are done
                if summary['pending'] == 0 and summary['running'] == 0:
                    print("🎉 All experiments completed!")
                    break
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                print("\n⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(60)  # Wait a bit before retrying
    
    def get_results_summary(self) -> Dict:
        """Get summary of all completed experiments"""
        return self.tracker.export_results_summary()

async def main():
    """Main execution function"""
    
    # Configuration: Set to True for testing with 1 dataset and 3 models
    TEST_MODE = True  # Change to False for full run
    FORCE_UPLOAD = True  # Set to True to re-upload all files even if they exist
    
    if TEST_MODE:
        print("🧪 RUNNING IN TEST MODE")
        print("=" * 50)
        print("📊 Testing with 1 dataset and 3 fast models")
        print("💡 Set TEST_MODE = False in main() for full run")
        print("=" * 50)
        launcher = CloudJobLauncher(test_mode=True)
    else:
        print("🚀 RUNNING IN FULL MODE")
        print("=" * 50)
        print("📊 Running all datasets with all models")
        print("=" * 50)
        launcher = CloudJobLauncher(test_mode=False)
    
    try:
        # Setup infrastructure with test mode settings
        if TEST_MODE:
            launcher.setup_infrastructure(max_datasets=1, force_upload=FORCE_UPLOAD)  # Use only 1 dataset for testing
        else:
            launcher.setup_infrastructure(force_upload=FORCE_UPLOAD)  # Use all datasets and models
        
        # Launch experiments with appropriate concurrency
        max_concurrent = 5 if TEST_MODE else 40
        await launcher.launch_all_experiments(max_concurrent=max_concurrent)
        
        # Monitor experiments
        launcher.monitor_experiments()
        
        # Get final results
        results = launcher.get_results_summary()
        print(f"📊 Final summary: {results['experiment_summary']}")
        
        if TEST_MODE:
            print("\n🎉 Test run completed successfully!")
            print("💡 Set TEST_MODE = False in main() to run all experiments")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 