#!/usr/bin/env python3
"""
Experiment Status Monitor - Check progress of running experiments
"""

import time
import json
from typing import Dict
from gcp_config import get_gcp_config
from experiment_tracker import ExperimentTracker
from gcp_storage import GCPStorageManager

def print_status_table(summary: Dict) -> None:
    """Print a nicely formatted status table"""
    print("\n" + "="*60)
    print("🔍 EXPERIMENT STATUS MONITOR")
    print("="*60)
    
    total = summary['total_jobs']
    pending = summary['pending']
    running = summary['running'] 
    completed = summary['completed']
    failed = summary['failed']
    cancelled = summary['cancelled']
    
    # Progress bar
    if total > 0:
        progress = (completed / total) * 100
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"Progress: [{bar}] {progress:.1f}% ({completed}/{total})")
    
    print(f"""
📊 Status Summary:
   ✅ Completed: {completed:>3}
   🔄 Running:   {running:>3}
   ⏳ Pending:   {pending:>3}
   ❌ Failed:    {failed:>3}
   🚫 Cancelled: {cancelled:>3}
   📋 Total:     {total:>3}
""")
    
    if summary.get('last_updated'):
        print(f"🕐 Last Updated: {summary['last_updated']}")

def print_detailed_jobs(tracker: ExperimentTracker) -> None:
    """Print detailed information about all jobs"""
    jobs = tracker._load_jobs()
    
    if not jobs:
        print("No jobs found.")
        return
    
    print("\n📋 DETAILED JOB STATUS")
    print("-" * 80)
    print(f"{'Job ID':<35} {'Model':<12} {'Dataset':<20} {'Status':<12}")
    print("-" * 80)
    
    for job in jobs:
        # Truncate long names for display
        job_display = job.job_id[:34] if len(job.job_id) > 34 else job.job_id
        dataset_display = job.dataset_name[:19] if len(job.dataset_name) > 19 else job.dataset_name
        
        # Status emoji
        status_emoji = {
            'pending': '⏳',
            'running': '🔄', 
            'completed': '✅',
            'failed': '❌',
            'cancelled': '🚫'
        }.get(job.status.value, '❓')
        
        print(f"{job_display:<35} {job.model_name:<12} {dataset_display:<20} {status_emoji} {job.status.value:<10}")
        
        # Show error message for failed jobs
        if job.status.value == 'failed' and job.error_message:
            error_short = job.error_message[:60] + "..." if len(job.error_message) > 60 else job.error_message
            print(f"    Error: {error_short}")

def download_results(storage: GCPStorageManager) -> None:
    """Download and show available results"""
    print("\n📥 DOWNLOADING RESULTS")
    print("-" * 40)
    
    try:
        # Check if results exist
        results_files = storage.list_files('results/raw/')
        
        if not results_files:
            print("No results available yet.")
            return
        
        print(f"Found {len(results_files)} result files:")
        for file_path in results_files:
            filename = file_path.split('/')[-1]
            print(f"  📄 {filename}")
        
        # Download experiment summary if available
        if storage.file_exists('results/experiment_summary.json'):
            summary = storage.download_json('results/experiment_summary.json')
            
            print(f"\n📊 Results Summary:")
            exp_summary = summary.get('experiment_summary', {})
            print(f"  Completed experiments: {exp_summary.get('completed', 0)}")
            print(f"  Total results files: {len(summary.get('completed_results', []))}")
            print(f"  Export timestamp: {summary.get('export_timestamp', 'Unknown')}")
            
            # Save locally
            with open('experiment_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"  💾 Downloaded to: experiment_summary.json")
        
    except Exception as e:
        print(f"❌ Error downloading results: {e}")

def main():
    """Main monitoring function"""
    try:
        config = get_gcp_config()
        tracker = ExperimentTracker(config)
        storage = GCPStorageManager(config)
        
        print("🚀 GCP ML Experiment Monitor")
        print("Press Ctrl+C to exit")
        
        while True:
            try:
                # Get current status
                summary = tracker.get_experiment_summary()
                print_status_table(summary)
                
                # Check if all done
                if summary['pending'] == 0 and summary['running'] == 0:
                    print("\n🎉 All experiments completed!")
                    
                    # Show detailed status
                    print_detailed_jobs(tracker)
                    
                    # Download results
                    download_results(storage)
                    
                    print("\n✅ Monitoring complete!")
                    break
                
                # Wait before next check
                print("\n⏳ Waiting 60 seconds for next check... (Ctrl+C to exit)")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n\n⏹️ Monitoring stopped by user")
                
                # Show final status
                summary = tracker.get_experiment_summary()
                print_status_table(summary)
                print_detailed_jobs(tracker)
                
                break
                
    except Exception as e:
        print(f"❌ Monitor error: {e}")
        raise

if __name__ == "__main__":
    main() 