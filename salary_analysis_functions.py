"""
Developer Salary Analysis - Refactored with Functions and Classes
================================================================

This module provides a comprehensive analysis of developer salaries using 
machine learning techniques. The code has been refactored to follow DRY principles
with proper functions, classes, and documentation.

Author: Data Science Student
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")


class DeveloperSalaryAnalyzer:
    """
    A comprehensive class for analyzing developer salary data.
    
    This class handles data generation, preprocessing, visualization, 
    and machine learning model training for developer salary analysis.
    """
    
    def __init__(self, n_samples=5000, random_state=42):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            n_samples (int): Number of sample records to generate
            random_state (int): Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.model_results = {}
        self.best_model_name = None
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Define constants
        self._setup_constants()
    
    def _setup_constants(self):
        """Set up constant values used throughout the analysis."""
        self.countries = ['United States', 'Germany', 'United Kingdom', 'Canada', 'Australia', 
                         'Netherlands', 'Sweden', 'France', 'Switzerland', 'India', 'Brazil', 
                         'Poland', 'Russia', 'Spain', 'Italy']
        
        self.dev_types = ['Full-stack developer', 'Back-end developer', 'Front-end developer', 
                         'Data scientist', 'DevOps engineer', 'Mobile developer', 'QA engineer',
                         'Data engineer', 'Machine learning engineer', 'Security engineer']
        
        self.education_levels = ['Bachelor\'s degree', 'Master\'s degree', 'Some college', 
                                'High school', 'PhD', 'Professional certificate']
        
        self.company_sizes = ['2-9 employees', '10-19 employees', '20-99 employees', 
                             '100-499 employees', '500-999 employees', '1,000-4,999 employees', 
                             '5,000-9,999 employees', '10,000+ employees']
        
        self._setup_multipliers()
    
    def _setup_multipliers(self):
        """Set up salary multipliers for different categories."""
        self.country_multiplier = {
            'United States': 1.4, 'Switzerland': 1.6, 'Germany': 1.1, 'United Kingdom': 1.2,
            'Canada': 1.1, 'Australia': 1.2, 'Netherlands': 1.3, 'Sweden': 1.2,
            'France': 1.0, 'India': 0.3, 'Brazil': 0.4, 'Poland': 0.6,
            'Russia': 0.5, 'Spain': 0.8, 'Italy': 0.8
        }
        
        self.role_multiplier = {
            'Machine learning engineer': 1.5, 'Data scientist': 1.4, 'Security engineer': 1.4,
            'Data engineer': 1.3, 'DevOps engineer': 1.3, 'Full-stack developer': 1.2,
            'Back-end developer': 1.15, 'Front-end developer': 1.0, 'Mobile developer': 1.1,
            'QA engineer': 0.9
        }
        
        self.education_bonus = {
            'PhD': 1.3, 'Master\'s degree': 1.15, 'Bachelor\'s degree': 1.0,
            'Professional certificate': 0.95, 'Some college': 0.85, 'High school': 0.75
        }
    
    def generate_sample_data(self):
        """
        Generate realistic synthetic developer salary data.
        
        Returns:
            pd.DataFrame: Generated dataset with developer information and salaries
        """
        print("Generating sample developer salary data...")
        
        # Generate basic demographic data
        data = self._generate_demographic_data()
        
        # Calculate salaries based on various factors
        salaries = self._calculate_salaries(data)
        data['ConvertedCompYearly'] = salaries
        
        # Create DataFrame and clean unrealistic combinations
        df = pd.DataFrame(data)
        df = self._clean_data(df)
        
        self.df = df
        
        print(f"Dataset created with {len(df)} records")
        print(f"Salary range: ${df['ConvertedCompYearly'].min():,} - ${df['ConvertedCompYearly'].max():,}")
        
        return df
    
    def _generate_demographic_data(self):
        """Generate demographic and professional data for developers."""
        return {
            'Country': np.random.choice(self.countries, self.n_samples, 
                                      p=[0.25, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.04, 0.03, 
                                        0.15, 0.06, 0.03, 0.03, 0.03, 0.04]),
            
            'DevType': np.random.choice(self.dev_types, self.n_samples,
                                      p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.02]),
            
            'YearsCodePro': np.random.exponential(scale=3, size=self.n_samples).astype(int),
            
            'EdLevel': np.random.choice(self.education_levels, self.n_samples,
                                      p=[0.45, 0.25, 0.12, 0.08, 0.06, 0.04]),
            
            'OrgSize': np.random.choice(self.company_sizes, self.n_samples,
                                      p=[0.08, 0.10, 0.18, 0.22, 0.12, 0.15, 0.08, 0.07]),
            
            'Remote': np.random.choice(['Fully remote', 'Hybrid', 'In-office'], self.n_samples,
                                     p=[0.35, 0.45, 0.20]),
            
            'Age': np.random.normal(32, 8, self.n_samples).astype(int),
            
            'DatabaseWorkedWith_count': np.random.poisson(3, self.n_samples),
            'LanguageWorkedWith_count': np.random.poisson(4, self.n_samples),
            'PlatformWorkedWith_count': np.random.poisson(2, self.n_samples),
            
            'AI_Tools_Used': np.random.choice([0, 1], self.n_samples, p=[0.35, 0.65]),
            'OpenSource_Contributor': np.random.choice([0, 1], self.n_samples, p=[0.60, 0.40])
        }
    
    def _calculate_salaries(self, data):
        """
        Calculate realistic salaries based on multiple factors.
        
        Args:
            data (dict): Dictionary containing developer characteristics
            
        Returns:
            list: List of calculated salaries
        """
        base_salary = 50000
        salaries = []
        
        for i in range(self.n_samples):
            salary = self._calculate_individual_salary(data, i, base_salary)
            salaries.append(int(salary))
        
        return salaries
    
    def _calculate_individual_salary(self, data, index, base_salary):
        """Calculate salary for a single developer based on their characteristics."""
        # Get multipliers for this individual
        country_mult = self.country_multiplier[data['Country'][index]]
        role_mult = self.role_multiplier[data['DevType'][index]]
        edu_mult = self.education_bonus[data['EdLevel'][index]]
        
        # Calculate experience and other factors
        experience_mult = 1 + (data['YearsCodePro'][index] * 0.05)
        age_factor = 1 + ((data['Age'][index] - 25) * 0.01)
        skills_bonus = 1 + (data['LanguageWorkedWith_count'][index] * 0.02)
        ai_bonus = 1.1 if data['AI_Tools_Used'][index] else 1.0
        opensource_bonus = 1.05 if data['OpenSource_Contributor'][index] else 1.0
        
        # Company size effect
        size_multiplier = {
            '2-9 employees': 0.8, '10-19 employees': 0.85,
            '20-99 employees': 0.9, '100-499 employees': 0.95,
            '500-999 employees': 1.0, '1,000-4,999 employees': 1.1,
            '5,000-9,999 employees': 1.15, '10,000+ employees': 1.2
        }
        
        size_mult = size_multiplier[data['OrgSize'][index]]
        
        # Remote work adjustment
        remote_mult = {'Fully remote': 1.05, 'Hybrid': 1.02, 'In-office': 1.0}[data['Remote'][index]]
        
        # Calculate final salary
        salary = (base_salary * country_mult * role_mult * edu_mult * 
                 experience_mult * age_factor * skills_bonus * size_mult * 
                 ai_bonus * opensource_bonus * remote_mult)
        
        # Add noise and constraints
        salary *= np.random.normal(1, 0.15)
        salary = max(25000, min(500000, salary))
        
        return salary
    
    def _clean_data(self, df):
        """Clean the dataset by removing unrealistic combinations."""
        df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]
        df = df[df['YearsCodePro'] <= 40]
        df = df[df['YearsCodePro'] <= df['Age'] - 16]
        return df.reset_index(drop=True)
    
    def perform_eda(self, save_plots=False):
        """
        Perform comprehensive Exploratory Data Analysis.
        
        Args:
            save_plots (bool): Whether to save plot images to files
        """
        if self.df is None:
            raise ValueError("Data not generated yet. Call generate_sample_data() first.")
        
        print("Performing Exploratory Data Analysis...")
        
        # Basic dataset information
        self._print_dataset_overview()
        
        # Generate visualizations
        self._create_salary_distribution_plots(save_plots)
        self._create_categorical_analysis_plots(save_plots)
        self._create_correlation_analysis(save_plots)
        
        print("EDA completed successfully!")
    
    def _print_dataset_overview(self):
        """Print basic information about the dataset."""
        print("=== DATASET OVERVIEW ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes}")
        print(f"\nMissing Values:\n{self.df.isnull().sum()}")
        print(f"\nBasic Statistics:\n{self.df.describe()}")
    
    def _create_salary_distribution_plots(self, save_plots=False):
        """Create and display salary distribution visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Salary histogram
        axes[0,0].hist(self.df['ConvertedCompYearly'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Annual Salaries', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Annual Salary ($)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(self.df['ConvertedCompYearly'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.df["ConvertedCompYearly"].mean():,.0f}')
        axes[0,0].axvline(self.df['ConvertedCompYearly'].median(), color='green', linestyle='--', 
                          label=f'Median: ${self.df["ConvertedCompYearly"].median():,.0f}')
        axes[0,0].legend()
        
        # Log-transformed salary
        axes[0,1].hist(np.log(self.df['ConvertedCompYearly']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Log-Transformed Salary Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Log(Annual Salary)')
        axes[0,1].set_ylabel('Frequency')
        
        # Experience vs Salary
        axes[1,0].scatter(self.df['YearsCodePro'], self.df['ConvertedCompYearly'], alpha=0.5, s=30)
        axes[1,0].set_title('Experience vs Salary', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Years of Professional Coding')
        axes[1,0].set_ylabel('Annual Salary ($)')
        
        # Age vs Salary
        axes[1,1].scatter(self.df['Age'], self.df['ConvertedCompYearly'], alpha=0.5, s=30, color='coral')
        axes[1,1].set_title('Age vs Salary', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Age')
        axes[1,1].set_ylabel('Annual Salary ($)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('salary_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_categorical_analysis_plots(self, save_plots=False):
        """Create visualizations for categorical feature analysis."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Salary by Country (top 10)
        country_salaries = self.df.groupby('Country')['ConvertedCompYearly'].median().sort_values(ascending=False).head(10)
        axes[0,0].bar(range(len(country_salaries)), country_salaries.values, color='steelblue')
        axes[0,0].set_title('Median Salary by Country (Top 10)', fontsize=12, fontweight='bold')
        axes[0,0].set_xticks(range(len(country_salaries)))
        axes[0,0].set_xticklabels(country_salaries.index, rotation=45, ha='right')
        axes[0,0].set_ylabel('Median Salary ($)')
        
        # Salary by Developer Type
        role_salaries = self.df.groupby('DevType')['ConvertedCompYearly'].median().sort_values(ascending=False)
        axes[0,1].bar(range(len(role_salaries)), role_salaries.values, color='darkorange')
        axes[0,1].set_title('Median Salary by Developer Type', fontsize=12, fontweight='bold')
        axes[0,1].set_xticks(range(len(role_salaries)))
        axes[0,1].set_xticklabels(role_salaries.index, rotation=45, ha='right')
        axes[0,1].set_ylabel('Median Salary ($)')
        
        # Salary by Education Level
        edu_salaries = self.df.groupby('EdLevel')['ConvertedCompYearly'].median().sort_values(ascending=False)
        axes[0,2].bar(range(len(edu_salaries)), edu_salaries.values, color='forestgreen')
        axes[0,2].set_title('Median Salary by Education Level', fontsize=12, fontweight='bold')
        axes[0,2].set_xticks(range(len(edu_salaries)))
        axes[0,2].set_xticklabels(edu_salaries.index, rotation=45, ha='right')
        axes[0,2].set_ylabel('Median Salary ($)')
        
        # Salary by Company Size
        size_order = ['2-9 employees', '10-19 employees', '20-99 employees', 
                      '100-499 employees', '500-999 employees', '1,000-4,999 employees',
                      '5,000-9,999 employees', '10,000+ employees']
        size_salaries = self.df.groupby('OrgSize')['ConvertedCompYearly'].median().reindex(size_order)
        axes[1,0].bar(range(len(size_salaries)), size_salaries.values, color='purple')
        axes[1,0].set_title('Median Salary by Company Size', fontsize=12, fontweight='bold')
        axes[1,0].set_xticks(range(len(size_salaries)))
        axes[1,0].set_xticklabels(size_salaries.index, rotation=45, ha='right')
        axes[1,0].set_ylabel('Median Salary ($)')
        
        # Salary by Remote Work
        remote_salaries = self.df.groupby('Remote')['ConvertedCompYearly'].median().sort_values(ascending=False)
        axes[1,1].bar(range(len(remote_salaries)), remote_salaries.values, color='teal')
        axes[1,1].set_title('Median Salary by Work Arrangement', fontsize=12, fontweight='bold')
        axes[1,1].set_xticks(range(len(remote_salaries)))
        axes[1,1].set_xticklabels(remote_salaries.index)
        axes[1,1].set_ylabel('Median Salary ($)')
        
        # AI Tools impact
        ai_salaries = self.df.groupby('AI_Tools_Used')['ConvertedCompYearly'].median()
        axes[1,2].bar(['No AI Tools', 'Uses AI Tools'], ai_salaries.values, color=['red', 'blue'])
        axes[1,2].set_title('Median Salary: AI Tools Usage', fontsize=12, fontweight='bold')
        axes[1,2].set_ylabel('Median Salary ($)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _create_correlation_analysis(self, save_plots=False):
        """Create correlation matrix visualization."""
        # Prepare numerical features for correlation
        df_corr = self.df.copy()
        
        # Encode categorical variables
        le_dict = {}
        categorical_cols = ['Country', 'DevType', 'EdLevel', 'OrgSize', 'Remote']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_corr[f'{col}_encoded'] = le.fit_transform(df_corr[col])
            le_dict[col] = le
        
        # Select numerical columns
        numerical_cols = ['YearsCodePro', 'Age', 'DatabaseWorkedWith_count', 'LanguageWorkedWith_count',
                         'PlatformWorkedWith_count', 'AI_Tools_Used', 'OpenSource_Contributor', 'ConvertedCompYearly']
        encoded_cols = [f'{col}_encoded' for col in categorical_cols]
        
        correlation_data = df_corr[numerical_cols + encoded_cols]
        
        # Create correlation matrix
        plt.figure(figsize=(14, 10))
        correlation_matrix = correlation_data.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Show strongest correlations with salary
        salary_correlations = correlation_matrix['ConvertedCompYearly'].abs().sort_values(ascending=False)[1:]
        print("\n=== STRONGEST CORRELATIONS WITH SALARY ===")
        for feature, corr in salary_correlations.head(10).items():
            print(f"{feature}: {corr:.3f}")
    
    def engineer_features(self):
        """
        Perform feature engineering to create new predictive features.
        
        Returns:
            pd.DataFrame: Dataset with engineered features
        """
        if self.df is None:
            raise ValueError("Data not generated yet. Call generate_sample_data() first.")
        
        print("Performing feature engineering...")
        
        df_ml = self.df.copy()
        
        # Create experience bins
        df_ml['Experience_Level'] = pd.cut(df_ml['YearsCodePro'], 
                                          bins=[0, 2, 5, 10, 20, 50], 
                                          labels=['Junior', 'Mid-level', 'Senior', 'Lead', 'Executive'])
        
        # Create age groups
        df_ml['Age_Group'] = pd.cut(df_ml['Age'], 
                                   bins=[0, 25, 35, 45, 100], 
                                   labels=['Young', 'Mid-career', 'Experienced', 'Veteran'])
        
        # Create total skills count
        df_ml['Total_Skills'] = (df_ml['DatabaseWorkedWith_count'] + 
                                df_ml['LanguageWorkedWith_count'] + 
                                df_ml['PlatformWorkedWith_count'])
        
        # Create high-paying countries indicator
        high_paying_countries = ['United States', 'Switzerland', 'Netherlands', 'Germany', 'Australia']
        df_ml['High_Paying_Country'] = df_ml['Country'].isin(high_paying_countries).astype(int)
        
        # Create high-demand roles indicator
        high_demand_roles = ['Machine learning engineer', 'Data scientist', 'Security engineer', 'Data engineer']
        df_ml['High_Demand_Role'] = df_ml['DevType'].isin(high_demand_roles).astype(int)
        
        # Create advanced degree indicator
        df_ml['Advanced_Degree'] = df_ml['EdLevel'].isin(['Master\'s degree', 'PhD']).astype(int)
        
        self.df = df_ml
        
        print("Feature engineering completed!")
        print(f"New features created: Total_Skills, High_Paying_Country, High_Demand_Role, Advanced_Degree")
        
        return df_ml
    
    def prepare_features_for_modeling(self):
        """
        Prepare features for machine learning modeling.
        
        Returns:
            tuple: (X, y) feature matrix and target variable
        """
        if self.df is None:
            raise ValueError("Data not prepared. Call generate_sample_data() and engineer_features() first.")
        
        print("Preparing features for modeling...")
        
        # Select features for the model
        feature_columns = ['YearsCodePro', 'Age', 'DatabaseWorkedWith_count', 'LanguageWorkedWith_count',
                          'PlatformWorkedWith_count', 'AI_Tools_Used', 'OpenSource_Contributor',
                          'Total_Skills', 'High_Paying_Country', 'High_Demand_Role', 'Advanced_Degree']
        
        # One-hot encode categorical features
        categorical_features = ['Country', 'DevType', 'EdLevel', 'OrgSize', 'Remote']
        df_encoded = pd.get_dummies(self.df, columns=categorical_features, prefix=categorical_features)
        
        # Get all feature columns (including encoded ones)
        encoded_feature_cols = [col for col in df_encoded.columns if any(cat in col for cat in categorical_features)]
        all_features = feature_columns + encoded_feature_cols
        
        # Prepare final dataset
        self.X = df_encoded[all_features]
        self.y = df_encoded['ConvertedCompYearly']
        
        print(f"Features prepared for modeling:")
        print(f"Number of features: {len(all_features)}")
        print(f"Feature matrix shape: {self.X.shape}")
        print(f"Target variable shape: {self.y.shape}")
        
        return self.X, self.y
    
    def train_models(self, test_size=0.2):
        """
        Train multiple machine learning models and evaluate performance.
        
        Args:
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Dictionary containing trained models and results
        """
        if self.X is None or self.y is None:
            raise ValueError("Features not prepared. Call prepare_features_for_modeling() first.")
        
        print("Training machine learning models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=self.random_state)
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Initialize models
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        }
        
        # Train and evaluate models
        self.model_results = {}
        
        for name, model in self.models.items():
            print(f"\n=== Training {name} ===")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            self.model_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'predictions': y_pred_test,
                'y_test': y_test
            }
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {test_r2:.4f}")
            print(f"Test RMSE: ${test_rmse:,.2f}")
            print(f"Test MAE: ${test_mae:,.2f}")
        
        # Select best model based on test R²
        self.best_model_name = max(self.model_results.keys(), key=lambda k: self.model_results[k]['test_r2'])
        
        print(f"\n=== BEST MODEL: {self.best_model_name} ===")
        print(f"Test R²: {self.model_results[self.best_model_name]['test_r2']:.4f}")
        print(f"Test RMSE: ${self.model_results[self.best_model_name]['test_rmse']:,.2f}")
        print(f"Test MAE: ${self.model_results[self.best_model_name]['test_mae']:,.2f}")
        
        return self.model_results
    
    def visualize_model_performance(self, save_plots=False):
        """
        Create visualizations of model performance.
        
        Args:
            save_plots (bool): Whether to save plot images to files
        """
        if not self.model_results:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Model performance comparison
        model_names = list(self.model_results.keys())
        r2_scores = [self.model_results[name]['test_r2'] for name in model_names]
        rmse_scores = [self.model_results[name]['test_rmse'] for name in model_names]
        
        axes[0,0].bar(model_names, r2_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0,0].set_title('Model R² Scores (Test Set)', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            axes[0,0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        axes[0,1].bar(model_names, rmse_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0,1].set_title('Model RMSE Scores (Test Set)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('RMSE ($)')
        for i, v in enumerate(rmse_scores):
            axes[0,1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')
        
        # Best model predictions vs actual
        best_predictions = self.model_results[self.best_model_name]['predictions']
        y_test = self.model_results[self.best_model_name]['y_test']
        
        axes[1,0].scatter(y_test, best_predictions, alpha=0.6, s=30)
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_title(f'{self.best_model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Actual Salary ($)')
        axes[1,0].set_ylabel('Predicted Salary ($)')
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6, s=30)
        axes[1,1].axhline(y=0, color='r', linestyle='--')
        axes[1,1].set_title(f'{self.best_model_name}: Residuals Plot', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Predicted Salary ($)')
        axes[1,1].set_ylabel('Residuals ($)')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_feature_importance(self, save_plots=False):
        """
        Analyze and visualize feature importance for tree-based models.
        
        Args:
            save_plots (bool): Whether to save plot images to files
        """
        if self.best_model_name not in ['Random Forest', 'Gradient Boosting']:
            print("Feature importance analysis only available for tree-based models.")
            return
        
        best_model = self.model_results[self.best_model_name]['model']
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 most important features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Most Important Features ({self.best_model_name})', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        if save_plots:
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n=== TOP 10 MOST IMPORTANT FEATURES ===")
        for i, row in feature_importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
    
    def generate_creative_insights(self):
        """Generate and display creative insights from the data analysis."""
        print("=== CREATIVE INSIGHTS FROM THE DATA ===")
        
        # AI tools impact analysis
        ai_impact = self.df.groupby('AI_Tools_Used')['ConvertedCompYearly'].agg(['mean', 'median', 'count'])
        ai_diff = ai_impact.loc[1, 'median'] - ai_impact.loc[0, 'median']
        print(f"\n1. AI TOOLS PREMIUM:")
        print(f"   Developers using AI tools earn ${ai_diff:,.0f} more (median)")
        print(f"   AI users median: ${ai_impact.loc[1, 'median']:,.0f}")
        print(f"   Non-AI users median: ${ai_impact.loc[0, 'median']:,.0f}")
        
        # Open source contribution impact
        os_impact = self.df.groupby('OpenSource_Contributor')['ConvertedCompYearly'].agg(['mean', 'median'])
        os_diff = os_impact.loc[1, 'median'] - os_impact.loc[0, 'median']
        print(f"\n2. OPEN SOURCE PREMIUM:")
        print(f"   Open source contributors earn ${os_diff:,.0f} more (median)")
        
        # High earner profile analysis
        self.df['Experience_to_Age_Ratio'] = self.df['YearsCodePro'] / self.df['Age']
        high_earners = self.df[self.df['ConvertedCompYearly'] > self.df['ConvertedCompYearly'].quantile(0.8)]
        print(f"\n3. HIGH EARNER PROFILE (Top 20%):")
        print(f"   Average age: {high_earners['Age'].mean():.1f} years")
        print(f"   Average experience: {high_earners['YearsCodePro'].mean():.1f} years")
        print(f"   Experience-to-age ratio: {high_earners['Experience_to_Age_Ratio'].mean():.2f}")
        
        # Skills diversity impact
        skills_salary = self.df.groupby(pd.cut(self.df['Total_Skills'], bins=5))['ConvertedCompYearly'].median()
        print(f"\n4. SKILLS DIVERSITY IMPACT:")
        print(f"   Salary increases with skill diversity:")
        for skill_range, salary in skills_salary.items():
            print(f"   {skill_range}: ${salary:,.0f}")
    
    def create_predictive_scenarios(self):
        """Create and analyze predictive scenarios using the trained model."""
        if self.best_model_name is None:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        print("=== CREATIVE PREDICTIVE SCENARIOS ===")
        print("\\nUsing our trained model to predict salaries for different developer profiles:\\n")
        
        # Define scenarios (this would need to be adapted to match your feature encoding)
        scenarios = {
            'The AI-Powered Fresh Graduate': {
                'YearsCodePro': 1, 'Age': 22, 'DatabaseWorkedWith_count': 3,
                'LanguageWorkedWith_count': 5, 'AI_Tools_Used': 1, 'OpenSource_Contributor': 1,
                # Additional features would need to be set based on your encoding
            },
            # Add other scenarios as needed
        }
        
        # This is a simplified version - full implementation would require
        # proper feature encoding for each scenario
        print("Scenario predictions would be implemented here with proper feature encoding.")
    
    def run_complete_analysis(self, save_plots=False):
        """
        Run the complete salary analysis pipeline.
        
        Args:
            save_plots (bool): Whether to save all generated plots as image files
        """
        print("Starting Complete Developer Salary Analysis...")
        print("=" * 60)
        
        # Step 1: Generate data
        self.generate_sample_data()
        
        # Step 2: Exploratory Data Analysis
        self.perform_eda(save_plots=save_plots)
        
        # Step 3: Feature Engineering
        self.engineer_features()
        
        # Step 4: Prepare features for modeling
        self.prepare_features_for_modeling()
        
        # Step 5: Train models
        self.train_models()
        
        # Step 6: Visualize model performance
        self.visualize_model_performance(save_plots=save_plots)
        
        # Step 7: Analyze feature importance
        self.analyze_feature_importance(save_plots=save_plots)
        
        # Step 8: Generate creative insights
        self.generate_creative_insights()
        
        print("\\n" + "=" * 60)
        print("Complete analysis finished successfully!")
        
        if save_plots:
            print("All visualizations have been saved as PNG files.")
        
        return {
            'dataset': self.df,
            'models': self.model_results,
            'best_model': self.best_model_name,
            'feature_matrix': self.X,
            'target': self.y
        }


def main():
    """Main function to run the developer salary analysis."""
    # Initialize the analyzer
    analyzer = DeveloperSalaryAnalyzer(n_samples=5000, random_state=42)
    
    # Run complete analysis with plot saving enabled
    results = analyzer.run_complete_analysis(save_plots=True)
    
    # Print final summary
    print("\\n=== FINAL ANALYSIS SUMMARY ===")
    print(f"Dataset size: {len(results['dataset'])} records")
    print(f"Best model: {results['best_model']}")
    print(f"Model accuracy: {results['models'][results['best_model']]['test_r2']:.3f}")
    print("All visualizations saved as PNG files for blog post use.")


if __name__ == "__main__":
    main()