#!/usr/bin/env python3
"""
Runner script for ML experiment statistical analysis
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements_file = "requirements_analysis.txt"
    if Path(requirements_file).exists():
        print("📦 Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        print("✅ Requirements installed successfully")
    else:
        print("⚠️  requirements_analysis.txt not found")

def check_gcp_config():
    """Check if GCP configuration files exist"""
    required_files = ["gcp_config.py", "gcp_storage.py"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure you have the GCP configuration files in the current directory")
        return False
    
    return True

def main():
    """Main runner function"""
    print("🚀 ML Experiment Statistical Analysis Runner")
    print("=" * 60)
    
    # Check if we should install requirements
    install_deps = input("Install requirements? (y/n): ").lower().strip() == 'y'
    
    if install_deps:
        try:
            install_requirements()
        except Exception as e:
            print(f"❌ Error installing requirements: {e}")
            return
    
    # Check GCP configuration
    if not check_gcp_config():
        return
    
    # Run the analysis
    print("\n🔍 Starting statistical analysis...")
    try:
        from stat_analysis import main as run_analysis
        run_analysis()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 