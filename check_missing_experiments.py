#!/usr/bin/env python3
"""
Check Missing Experiments - Analyze which dataset+model combinations didn't execute or failed
"""

import pandas as pd
import asyncio
import uuid
from pathlib import Path
from gcp_config import get_gcp_config, ExperimentJob, ExperimentStatus
from gcp_storage import GCPStorageManager

def get_expected_combinations():
    """Get all expected dataset+model combinations from local data"""
    # Read datasets summary
    datasets_summary = pd.read_csv('data/split/datasets_summary.csv')
    
    # Models used in full mode (from cloud_job_launcher.py)
    models = ['XGBoost', 'LightGBM', 'CatBoost', 'EBM', 'AutoSklearn', 'AutoGluon']
    
    expected_combinations = []
    for _, row in datasets_summary.iterrows():
        dataset_name = f"dataset_{row['dataset_id']}_{row['name']}"
        for model_name in models:
            expected_combinations.append({
                'dataset_id': row['dataset_id'],
                'dataset_name': dataset_name,
                'model_name': model_name,
                'combination': f"{model_name}_{dataset_name}",
                'n_features': row['n_features'],
                'n_classes': row['n_classes'],
                'train_samples': row['train_samples'],
                'test_samples': row['test_samples']
            })
    
    return expected_combinations

def examine_bucket_files():
    """Examine actual files in the bucket to understand naming pattern"""
    config = get_gcp_config()
    storage = GCPStorageManager(config)
    
    try:
        result_files = storage.list_files("results/raw/")
        print(f"📁 Found {len(result_files)} files in results/raw/")
        
        if result_files:
            print("\n📋 Sample filenames (first 10):")
            for i, file_path in enumerate(result_files[:10]):
                filename = Path(file_path).name
                print(f"   {i+1:2d}. {filename}")
            
            if len(result_files) > 10:
                print(f"   ... and {len(result_files) - 10} more files")
                
            # Show some more samples to understand pattern
            print(f"\n📋 Last 5 filenames:")
            for i, file_path in enumerate(result_files[-5:]):
                filename = Path(file_path).name
                print(f"   {len(result_files)-4+i:2d}. {filename}")
        
        return result_files
        
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return []

def parse_actual_results(result_files):
    """Parse actual results from bucket files with improved logic"""
    actual_results = []
    unparsed_files = []
    
    for file_path in result_files:
        filename = Path(file_path).stem  # Remove .json extension
        
        # Try different parsing strategies based on common patterns
        parsed = False
        
        # Strategy 1: ModelName_dataset_id_name format
        if "_dataset_" in filename:
            parts = filename.split("_dataset_", 1)
            if len(parts) == 2:
                model_name = parts[0]
                dataset_part = parts[1]
                dataset_name = f"dataset_{dataset_part}"
                
                actual_results.append({
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'combination': f"{model_name}_{dataset_name}",
                    'file_path': file_path
                })
                parsed = True
        
        # Strategy 2: ModelName_datasetname format (no "dataset_" prefix)
        if not parsed:
            # Common model names
            model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'EBM', 'AutoSklearn', 'AutoGluon']
            
            for model in model_names:
                if filename.startswith(model + "_"):
                    dataset_part = filename[len(model) + 1:]  # Remove "ModelName_"
                    
                    # Check if this looks like a dataset name
                    if dataset_part:
                        # If it doesn't already start with "dataset_", add it
                        if not dataset_part.startswith("dataset_"):
                            dataset_name = f"dataset_{dataset_part}"
                        else:
                            dataset_name = dataset_part
                            
                        actual_results.append({
                            'model_name': model,
                            'dataset_name': dataset_name,
                            'combination': f"{model}_{dataset_name}",
                            'file_path': file_path
                        })
                        parsed = True
                        break
        
        if not parsed:
            unparsed_files.append(filename)
    
    print(f"\n✅ Successfully parsed: {len(actual_results)} files")
    if unparsed_files:
        print(f"❌ Could not parse: {len(unparsed_files)} files")
        print("📋 Unparsed files (first 10):")
        for filename in unparsed_files[:10]:
            print(f"   {filename}")
    
    return actual_results

def get_actual_results():
    """Get actual results from GCP bucket with improved parsing"""
    result_files = examine_bucket_files()
    
    if not result_files:
        return []
    
    return parse_actual_results(result_files)

def parse_user_selection(user_input, max_number):
    """Parse user input to get selected experiment numbers"""
    selected_numbers = []
    
    if not user_input.strip():
        return selected_numbers
    
    # Split by comma and process each part
    parts = user_input.split(',')
    
    for part in parts:
        part = part.strip()
        
        # Handle ranges like "1-3"
        if '-' in part:
            try:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                
                if 1 <= start_num <= max_number and 1 <= end_num <= max_number and start_num <= end_num:
                    selected_numbers.extend(range(start_num, end_num + 1))
                else:
                    print(f"⚠️  Invalid range: {part} (must be between 1 and {max_number})")
            except ValueError:
                print(f"⚠️  Invalid range format: {part}")
        else:
            # Handle single numbers
            try:
                num = int(part)
                if 1 <= num <= max_number:
                    selected_numbers.append(num)
                else:
                    print(f"⚠️  Invalid number: {num} (must be between 1 and {max_number})")
            except ValueError:
                print(f"⚠️  Invalid number format: {part}")
    
    # Remove duplicates and sort
    selected_numbers = sorted(list(set(selected_numbers)))
    return selected_numbers

async def run_selected_combinations(selected_combinations, expected_combinations):
    """Run the selected combinations using CloudJobLauncher"""
    
    # Import here to avoid issues if not available
    try:
        from cloud_job_launcher import CloudJobLauncher
    except ImportError:
        print("❌ CloudJobLauncher not available. Make sure cloud_job_launcher.py is in the current directory.")
        return
    
    print(f"\n🚀 Preparing to run {len(selected_combinations)} selected experiments...")
    
    # Create launcher
    launcher = CloudJobLauncher(test_mode=False)
    
    # Create ExperimentJob objects for selected combinations
    selected_jobs = []
    for combo in selected_combinations:
        # Find the expected combination details
        combo_details = None
        for exp_combo in expected_combinations:
            if exp_combo['combination'] == combo:
                combo_details = exp_combo
                break
        
        if combo_details:
            job_id = f"{combo_details['model_name']}_{combo_details['dataset_name']}_{uuid.uuid4().hex[:8]}"
            job = ExperimentJob(
                job_id=job_id,
                model_name=combo_details['model_name'],
                dataset_name=combo_details['dataset_name'],
                status=ExperimentStatus.PENDING
            )
            selected_jobs.append(job)
            print(f"   📝 Created job: {job.job_id}")
    
    if not selected_jobs:
        print("❌ No valid jobs to run")
        return
    
    print(f"\n🔧 Launching {len(selected_jobs)} selected experiments...")
    
    # Launch jobs one by one with some delay
    for i, job in enumerate(selected_jobs, 1):
        try:
            print(f"\n[{i}/{len(selected_jobs)}] 🚀 Launching {job.model_name} on {job.dataset_name}")
            
            # Create Cloud Run job
            cloud_run_job_name = launcher.create_cloud_run_job(
                job.job_id,
                job.model_name, 
                job.dataset_name
            )
            
            # Small delay before execution
            await asyncio.sleep(0.01)
            
            # Execute the job
            launcher.execute_job(cloud_run_job_name)
            
            print(f"   ✅ Successfully launched: {cloud_run_job_name}")
            
            # Delay between jobs to avoid overwhelming the system
            if i < len(selected_jobs):
                print(f"   ⏳ Waiting 5 seconds before next job...")
                await asyncio.sleep(1)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"   ❌ Failed to launch {job.job_id}: {e}")
            continue
    
    print(f"\n🎉 Finished launching selected experiments!")
    print(f"💡 You can monitor progress with: gcloud run jobs list --region=us-central1")

def main():
    """Main analysis function"""
    print("🔍 Checking Missing/Failed Experiments")
    print("=" * 50)
    
    # Get expected combinations
    expected_combinations = get_expected_combinations()
    print(f"📊 Expected combinations: {len(expected_combinations)}")
    print(f"   - {len(set(combo['dataset_name'] for combo in expected_combinations))} datasets")
    print(f"   - {len(set(combo['model_name'] for combo in expected_combinations))} models")
    
    # Get actual results
    actual_results = get_actual_results()
    print(f"\n✅ Actual results found: {len(actual_results)}")
    
    if actual_results:
        print("\n📋 Sample actual results:")
        for i, result in enumerate(actual_results[:5]):
            print(f"   {i+1}. {result['combination']}")
        if len(actual_results) > 5:
            print(f"   ... and {len(actual_results) - 5} more")
    
    # Create analysis
    expected_combinations_set = set(combo['combination'] for combo in expected_combinations)
    actual_results_set = set(result['combination'] for result in actual_results)
    
    print(f"\n📊 Expected vs Actual:")
    print(f"   Expected: {len(expected_combinations_set)} combinations")
    print(f"   Found: {len(actual_results_set)} combinations")
    
    # Missing experiments (no results)
    missing_combinations = expected_combinations_set - actual_results_set
    print(f"   Missing: {len(missing_combinations)} combinations")
    
    # Show what we actually found vs expected
    if actual_results:
        print(f"\n📈 Success Rate: {len(actual_results_set)/len(expected_combinations_set)*100:.1f}%")
        
        # Group found results by model
        actual_results_df = pd.DataFrame(actual_results)
        print(f"\n📊 Found results by model:")
        model_counts = actual_results_df['model_name'].value_counts()
        for model, count in model_counts.items():
            print(f"   {model}: {count}")
        
        # Show which datasets have results
        datasets_with_results = set()
        for result in actual_results:
            datasets_with_results.add(result['dataset_name'])
        
        expected_datasets = set(combo['dataset_name'] for combo in expected_combinations)
        missing_datasets = expected_datasets - datasets_with_results
        
        print(f"\n📊 Dataset coverage:")
        print(f"   Datasets with results: {len(datasets_with_results)}")
        print(f"   Datasets missing all results: {len(missing_datasets)}")
        
        if missing_datasets:
            print(f"   Missing datasets:")
            for dataset in sorted(missing_datasets):
                print(f"     - {dataset}")
    
    if missing_combinations:
        print(f"\n❌ Missing combinations:")
        missing_list = sorted(list(missing_combinations))
        
        # Show enumerated list of missing combinations
        for i, combo in enumerate(missing_list, 1):
            print(f"   {i:2d}. {combo}")
        
        # Create detailed report
        missing_details = []
        for combo in expected_combinations:
            if combo['combination'] in missing_combinations:
                missing_details.append({
                    'combination': combo['combination'],
                    'dataset_id': combo['dataset_id'],
                    'model': combo['model_name'],
                    'dataset': combo['dataset_name']
                })
        
        missing_df = pd.DataFrame(missing_details)
        
        # Group by model
        print(f"\n📊 Missing combinations by model:")
        model_counts = missing_df['model'].value_counts()
        for model, count in model_counts.items():
            print(f"   {model}: {count}")
        
        # Save detailed report
        missing_df.to_csv('missing_experiments_report.csv', index=False)
        print(f"\n💾 Detailed report saved to: missing_experiments_report.csv")
        
        # Ask user to select specific experiments to run
        print(f"\n🤔 Which experiments would you like to run?")
        print(f"   Enter numbers separated by commas (e.g., '1,3,5' or '1-3,5' for ranges)")
        print(f"   Or press Enter to skip: ", end="")
        
        user_input = input().strip()
        
        if user_input:
            selected_numbers = parse_user_selection(user_input, len(missing_list))
            
            if selected_numbers:
                selected_combinations = [missing_list[i-1] for i in selected_numbers]
                print(f"\n✅ Selected {len(selected_combinations)} experiments to run:")
                for i, combo in enumerate(selected_combinations, 1):
                    print(f"   {i}. {combo}")
                
                # Confirm before running
                print(f"\n🔥 Are you sure you want to launch these {len(selected_combinations)} experiments? (y/n): ", end="")
                confirm = input().strip().lower()
                
                if confirm in ['y', 'yes']:
                    print(f"🚀 Starting execution of selected experiments...")
                    asyncio.run(run_selected_combinations(selected_combinations, expected_combinations))
                else:
                    print(f"⏭️  Cancelled execution.")
            else:
                print(f"❌ No valid experiments selected.")
        else:
            print(f"⏭️  Skipping execution of missing experiments.")
    
    # Check for unexpected results
    unexpected_results = actual_results_set - expected_combinations_set
    if unexpected_results:
        print(f"\n⚠️  Unexpected results: {len(unexpected_results)}")
        for unexpected in sorted(list(unexpected_results))[:10]:
            print(f"   {unexpected}")

if __name__ == "__main__":
    main() 