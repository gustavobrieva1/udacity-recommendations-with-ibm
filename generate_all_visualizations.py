#!/usr/bin/env python3
"""
Complete visualization generation script for the developer salary analysis blog post.
"""

import sys
import os
sys.path.append('/home/ghost2077/claude-projects/udacity_course')

from salary_analysis_functions import DeveloperSalaryAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def main():
    """Generate and save all visualizations for the blog post."""
    print("=" * 60)
    print("GENERATING DEVELOPER SALARY ANALYSIS VISUALIZATIONS")
    print("=" * 60)
    
    # Initialize analyzer with reasonable dataset size
    print("\n1. Initializing Developer Salary Analyzer...")
    analyzer = DeveloperSalaryAnalyzer(n_samples=2000, random_state=42)
    
    print("2. Generating synthetic dataset...")
    analyzer.generate_sample_data()
    
    print("3. Performing EDA and saving all plots...")
    analyzer.perform_eda(save_plots=True)
    
    print("4. Generating correlation matrix...")
    analyzer.plot_correlation_matrix(save_plot=True)
    
    # Generate additional insights visualization
    print("5. Generating salary insights by key factors...")
    try:
        analyzer.generate_salary_insights()
    except Exception as e:
        print(f"   Note: Insights generation skipped due to: {e}")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION GENERATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files for your blog post:")
    
    # List all PNG files in the directory
    png_files = [f for f in os.listdir('.') if f.endswith('.png')]
    if png_files:
        for i, file in enumerate(sorted(png_files), 1):
            print(f"  {i}. {file}")
    else:
        print("  No PNG files found in current directory")
    
    print(f"\nFiles saved in: {os.getcwd()}")
    print("You can now insert these images into your blog post!")

if __name__ == "__main__":
    main()