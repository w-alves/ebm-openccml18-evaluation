#!/usr/bin/env python3
"""
GCP Cloud Storage utilities for ML experiments
"""

import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from google.cloud import storage
from gcp_config import GCPConfig, get_gcp_config

class GCPStorageManager:
    """Manages Cloud Storage operations for ML experiments"""
    
    def __init__(self, config: Optional[GCPConfig] = None):
        self.config = config or get_gcp_config()
        self.client = storage.Client(project=self.config.project_id)
        self.bucket = self.client.bucket(self.config.bucket_name)
    
    def setup_bucket(self) -> None:
        """Create bucket if it doesn't exist"""
        try:
            self.bucket.reload()
            print(f"✅ Bucket {self.config.bucket_name} already exists")
        except Exception:
            print(f"📦 Creating bucket {self.config.bucket_name}")
            self.bucket = self.client.create_bucket(
                self.config.bucket_name,
                location=self.config.region
            )
            print(f"✅ Created bucket {self.config.bucket_name}")
    
    def file_needs_upload(self, local_path: str, remote_path: str) -> bool:
        """Check if local file needs to be uploaded (compares size and modification time)"""
        local_path = Path(local_path)
        if not local_path.exists():
            return False
            
        blob = self.bucket.blob(remote_path)
        if not blob.exists():
            return True
            
        # Reload blob to get current metadata
        blob.reload()
        
        # Compare file size
        local_size = local_path.stat().st_size
        remote_size = blob.size
        
        if local_size != remote_size:
            return True
            
        # Compare modification time (with some tolerance for clock differences)
        import datetime
        local_mtime = datetime.datetime.fromtimestamp(local_path.stat().st_mtime, tz=datetime.timezone.utc)
        remote_mtime = blob.updated
        
        # If local file is newer by more than 1 minute, upload it
        if local_mtime > remote_mtime + datetime.timedelta(minutes=1):
            return True
            
        return False
    
    def upload_directory(self, local_path: str, remote_prefix: str, force: bool = False) -> None:
        """Upload entire directory to Cloud Storage (skips already uploaded files unless force=True)"""
        local_path = Path(local_path)
        uploaded_count = 0
        skipped_count = 0
        
        print(f"📁 Checking directory: {local_path} -> gs://{self.config.bucket_name}/{remote_prefix}")
        
        for file_path in local_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                remote_path = f"{remote_prefix}/{relative_path}".replace('\\', '/')
                
                if force or self.file_needs_upload(str(file_path), remote_path):
                    blob = self.bucket.blob(remote_path)
                    blob.upload_from_filename(str(file_path))
                    print(f"📤 Uploaded {file_path} -> gs://{self.config.bucket_name}/{remote_path}")
                    uploaded_count += 1
                else:
                    print(f"⏭️  Skipped {file_path} (already uploaded)")
                    skipped_count += 1
        
        print(f"✅ Directory sync complete: {uploaded_count} uploaded, {skipped_count} skipped")
    
    def upload_file(self, local_path: str, remote_path: str, force: bool = False) -> None:
        """Upload single file to Cloud Storage (skips if already uploaded unless force=True)"""
        if force or self.file_needs_upload(local_path, remote_path):
            blob = self.bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
            print(f"📤 Uploaded {local_path} -> gs://{self.config.bucket_name}/{remote_path}")
        else:
            print(f"⏭️  Skipped {local_path} (already uploaded)")
    
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download single file from Cloud Storage"""
        blob = self.bucket.blob(remote_path)
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        blob.download_to_filename(local_path)
        print(f"📥 Downloaded gs://{self.config.bucket_name}/{remote_path} -> {local_path}")
    
    def download_directory(self, remote_prefix: str, local_path: str) -> None:
        """Download directory from Cloud Storage"""
        local_path = Path(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        
        blobs = self.bucket.list_blobs(prefix=remote_prefix)
        
        for blob in blobs:
            if not blob.name.endswith('/'):  # Skip directory markers
                relative_path = blob.name[len(remote_prefix):].lstrip('/')
                if relative_path:  # Skip empty paths
                    file_path = local_path / relative_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(file_path))
                    print(f"📥 Downloaded {blob.name} -> {file_path}")
    
    def list_files(self, prefix: str = "") -> List[str]:
        """List files in bucket with given prefix"""
        blobs = self.bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs if not blob.name.endswith('/')]
    
    def file_exists(self, remote_path: str) -> bool:
        """Check if file exists in Cloud Storage"""
        blob = self.bucket.blob(remote_path)
        return blob.exists()
    
    def directory_exists(self, remote_prefix: str) -> bool:
        """Check if directory exists and has files in Cloud Storage"""
        blobs = list(self.bucket.list_blobs(prefix=remote_prefix, max_results=1))
        return len(blobs) > 0
    
    def upload_json(self, data: Dict, remote_path: str, force: bool = False) -> None:
        """Upload JSON data to Cloud Storage (skips if content is same unless force=True)"""
        json_content = json.dumps(data, indent=2)
        
        if not force and self.file_exists(remote_path):
            # Check if content is different
            try:
                existing_content = self.bucket.blob(remote_path).download_as_text()
                if existing_content.strip() == json_content.strip():
                    print(f"⏭️  Skipped JSON {remote_path} (content unchanged)")
                    return
            except:
                pass  # If we can't read existing content, upload anyway
        
        blob = self.bucket.blob(remote_path)
        blob.upload_from_string(json_content, content_type='application/json')
        print(f"📤 Uploaded JSON -> gs://{self.config.bucket_name}/{remote_path}")
    
    def download_json(self, remote_path: str) -> Dict:
        """Download and parse JSON from Cloud Storage"""
        blob = self.bucket.blob(remote_path)
        content = blob.download_as_text()
        return json.loads(content)
    
    def get_signed_url(self, remote_path: str, expiration_hours: int = 1) -> str:
        """Get signed URL for downloading file"""
        from datetime import timedelta
        
        blob = self.bucket.blob(remote_path)
        url = blob.generate_signed_url(
            expiration=timedelta(hours=expiration_hours),
            method='GET'
        )
        return url

def prepare_datasets_for_cloud(storage_manager: GCPStorageManager, force_upload: bool = False) -> List[Dict]:
    """Prepare and upload datasets to Cloud Storage"""
    print("📊 Preparing datasets for cloud...")
    
    # Check if datasets are already uploaded
    if not force_upload and storage_manager.directory_exists('datasets'):
        print("📋 Checking if datasets are already uploaded...")
        
        # Check if we have the metadata file
        if storage_manager.file_exists('metadata/experiment_config.json'):
            try:
                config = storage_manager.download_json('metadata/experiment_config.json')
                print(f"✅ Found existing datasets: {len(config.get('datasets', []))} datasets available")
                print("💡 Use force_upload=True to re-upload all files")
                return config.get('datasets', [])
            except:
                print("⚠️  Could not read existing metadata, will re-upload")
    
    # Upload datasets directory
    if Path('data/split').exists():
        storage_manager.upload_directory('data/split', 'datasets', force=force_upload)
    else:
        raise FileNotFoundError("No datasets found! Run data preparation first.")
    
    # Upload models directory
    if Path('models').exists():
        storage_manager.upload_directory('models', 'code/models', force=force_upload)
    
    # Upload other necessary files
    files_to_upload = [
        'requirements_python38.txt',
        'requirements_python311.txt',
        'logging.conf'
    ]
    
    for file_path in files_to_upload:
        if Path(file_path).exists():
            storage_manager.upload_file(file_path, f'code/{file_path}', force=force_upload)
    
    # Get dataset information
    try:
        import pandas as pd
        datasets_summary = pd.read_csv('data/split/datasets_summary.csv')
        
        available_datasets = []
        for _, row in datasets_summary.iterrows():
            dataset_name = f"dataset_{row['dataset_id']}_{row['name']}"
            available_datasets.append({
                'name': dataset_name,
                'train_path': f'datasets/{dataset_name}_train.pkl',
                'test_path': f'datasets/{dataset_name}_test.pkl',
                'n_features': int(row['n_features']),
                'n_classes': int(row['n_classes'])
            })
        
        # Upload dataset metadata
        storage_manager.upload_json({
            'datasets': available_datasets,
            'models': ['AutoSklearn', 'AutoGluon', 'XGBoost', 'LightGBM', 'CatBoost', 'EBM'],
            'upload_timestamp': pd.Timestamp.now().isoformat()
        }, 'metadata/experiment_config.json', force=force_upload)
        
        print(f"✅ Prepared {len(available_datasets)} datasets for cloud execution")
        return available_datasets
        
    except Exception as e:
        print(f"❌ Error preparing datasets: {e}")
        return [] 