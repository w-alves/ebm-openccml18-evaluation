#!/usr/bin/env python3
# Quick single-experiment test
import sys
import asyncio
from run_experiments_optimized import run_optimized_experiment, load_environment_config, get_available_datasets
import tempfile
from pathlib import Path

async def quick_test():
    env_configs = load_environment_config()
    datasets = get_available_datasets()
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    # Test with first dataset and XGBoost
    dataset = datasets[0]
    env_config = env_configs['modern']  # Use modern environment
    temp_dir = Path(tempfile.mkdtemp())
    
    print(f"🧪 Testing: XGBoost on {dataset['name']}")
    result = await run_optimized_experiment('XGBoost', dataset, env_config, temp_dir)
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(quick_test())
