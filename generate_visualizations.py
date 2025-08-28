#!/usr/bin/env python3
"""
Script to generate visualizations for the developer salary analysis blog post.
"""

import sys
import os
sys.path.append('/home/ghost2077/claude-projects/udacity_course')

from salary_analysis_functions import DeveloperSalaryAnalyzer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def main():
    """Generate and save all visualizations for the blog post."""
    print("Initializing Developer Salary Analyzer...")
    
    # Initialize analyzer with smaller dataset for faster execution
    analyzer = DeveloperSalaryAnalyzer(n_samples=1000, random_state=42)
    
    print("Generating sample data...")
    analyzer.generate_sample_data()
    
    print("Performing EDA and saving plots...")
    analyzer.perform_eda(save_plots=True)
    
    print("Preparing features for modeling...")
    analyzer.prepare_features_for_modeling()
    
    print("Training model and generating feature importance plot...")
    analyzer.train_models()
    
    print("Generating correlation matrix...")
    analyzer.plot_correlation_matrix(save_plot=True)
    
    print("All visualizations have been generated and saved!")
    print("Check the current directory for PNG files to include in your blog post.")

if __name__ == "__main__":
    main()