#!/usr/bin/env python3
"""
Display information about generated visualizations for the blog post.
"""

import os
from pathlib import Path

def main():
    """Show all generated visualization files."""
    print("=" * 70)
    print("DEVELOPER SALARY ANALYSIS - GENERATED VISUALIZATIONS")
    print("=" * 70)
    
    current_dir = Path('.')
    png_files = list(current_dir.glob('*.png'))
    
    if png_files:
        print(f"\n✅ Found {len(png_files)} visualization files ready for your blog post:")
        print()
        
        for i, file in enumerate(sorted(png_files), 1):
            file_size = file.stat().st_size / 1024  # KB
            print(f"  {i}. {file.name}")
            print(f"     Size: {file_size:.1f} KB")
            print(f"     Path: {file.absolute()}")
            print()
        
        print("=" * 70)
        print("FILES ARE READY FOR BLOG POST INSERTION!")
        print("=" * 70)
        print("\nDescription of each visualization:")
        print("• categorical_analysis.png - Analysis of categorical variables vs salary")
        print("• correlation_matrix.png - Correlation heatmap of all numerical features")  
        print("• salary_distribution_analysis.png - Salary distribution and key statistics")
        
    else:
        print("\n❌ No PNG files found in the current directory.")
        print("Run the visualization generation script first.")
    
    print(f"\nWorking directory: {os.getcwd()}")

if __name__ == "__main__":
    main()