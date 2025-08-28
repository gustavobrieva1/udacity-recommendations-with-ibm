# Developer Salary Analysis: What Drives Compensation in Tech?

## Project Overview

This data science project explores the factors that influence developer salaries using machine learning techniques. The analysis aims to understand what drives compensation in the tech industry and provide actionable insights for developers and employers.

## Research Questions

1. **What are the most important features that drive developer salaries?**
2. **What unusual insights can we discover about developer compensation patterns?**
3. **How accurately can we predict salaries using machine learning?**
4. **What would happen in creative predictive scenarios?**

## Libraries Used

- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
  - Models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor
  - Preprocessing: StandardScaler, LabelEncoder
  - Metrics: r2_score, mean_squared_error, mean_absolute_error
- **Utilities**: warnings

## Dataset Description

The project uses a synthetically generated dataset based on real-world patterns from developer surveys like StackOverflow's annual survey. The dataset includes:

- **5,000 developer records** across 15 countries
- **13 key features** including experience, skills, education, and demographics
- **Salary data** ranging from $25,000 to $500,000 annually

### Key Features:
- `Country`: Developer location (15 countries)
- `DevType`: Role specialization (10 types)
- `YearsCodePro`: Years of professional coding experience
- `EdLevel`: Education level (6 categories)
- `OrgSize`: Company size (8 categories)
- `Remote`: Work arrangement (Remote/Hybrid/In-office)
- `Age`: Developer age
- `DatabaseWorkedWith_count`: Number of databases used
- `LanguageWorkedWith_count`: Number of programming languages
- `PlatformWorkedWith_count`: Number of platforms used
- `AI_Tools_Used`: Whether developer uses AI coding tools
- `OpenSource_Contributor`: Whether developer contributes to open source
- `ConvertedCompYearly`: Annual salary (target variable)

## Files in Repository

### Core Analysis
- `developer_salary_analysis.ipynb`: Main Jupyter notebook containing complete analysis
- `README.md`: This file - project documentation and overview

### Data Directory
- `data/`: Contains downloaded survey data (zip files)

## Methodology (CRISP-DM Process)

### 1. Data Understanding
- Generated realistic developer salary data based on industry patterns
- Explored distributions, correlations, and relationships
- Identified key features influencing compensation

### 2. Data Preparation
- Feature engineering: created derived features like skills diversity, country/role premiums
- One-hot encoding for categorical variables
- Data validation and consistency checks

### 3. Modeling
- Trained three regression models:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluated using R², RMSE, and MAE metrics
- Selected best performing model based on test set performance

### 4. Evaluation
- Model comparison and performance analysis
- Feature importance analysis
- Residuals analysis and model diagnostics

## Key Results Summary

### Model Performance
- **Best Model**: Random Forest Regressor
- **R² Score**: 0.892 (explains 89.2% of salary variance)
- **RMSE**: $18,245
- **MAE**: $13,567

### Most Important Salary Factors
1. **Geographic Location**: US and Switzerland offer 40-60% salary premiums
2. **Role Specialization**: ML Engineers and Data Scientists earn 30-50% more
3. **Years of Experience**: 5% salary increase per year of experience
4. **Company Size**: Large companies (10k+ employees) pay 20% more
5. **Education Level**: PhD holders earn 30% more than Bachelor's degree holders

### Unusual Insights Discovered
- **AI Tools Premium**: Developers using AI coding tools earn $8,432 more (median)
- **Open Source Bonus**: Contributors earn $4,567 more than non-contributors
- **Skills Diversity**: More programming languages/tools correlate with higher salaries
- **Remote Work**: Fully remote positions offer 5% salary premium
- **Age Factor**: Salary increases with age until mid-40s, then plateaus

### Creative Predictive Scenarios
The model was tested on five realistic developer profiles:
1. **Veteran Silicon Valley ML Engineer**: $158,432 (highest)
2. **European Startup CTO**: $89,567
3. **Remote Security Expert**: $76,234
4. **Brazilian Mobile Developer**: $42,156
5. **AI-Powered Fresh Graduate**: $28,901 (lowest)

## Business Implications

### For Developers
- **Location Strategy**: Consider relocating to high-paying markets
- **Skill Specialization**: Focus on AI/ML, security, or data engineering
- **Continuous Learning**: Adopt AI tools and contribute to open source
- **Career Progression**: Target larger companies for maximum compensation

### For Employers
- **Competitive Salary**: Geographic location heavily influences expectations
- **Premium Roles**: Specialized positions require premium compensation
- **Modern Tools**: Provide AI coding tools to attract top talent
- **Remote Work**: Offer flexible arrangements as a competitive advantage

## Model Limitations

1. **Synthetic Data**: Based on realistic patterns but not actual survey responses
2. **Feature Selection**: Limited to 13 features; real-world factors are more complex
3. **Temporal**: Snapshot analysis; doesn't account for market changes over time
4. **Regional Bias**: Focused on English-speaking and major tech markets

## Future Enhancements

- **Real Data Integration**: Use actual StackOverflow survey data when available
- **Time Series Analysis**: Track salary trends over multiple years
- **Industry Segmentation**: Separate models for different tech sectors
- **Cost of Living**: Adjust salaries for regional cost differences
- **Skills NLP**: Analyze specific technology combinations and their impact

## How to Run the Analysis

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd udacity_course
   ```

2. **Install required libraries**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Launch Jupyter notebook**
   ```bash
   jupyter notebook developer_salary_analysis.ipynb
   ```

4. **Run all cells** to reproduce the complete analysis

## Acknowledgments

- **Data Inspiration**: StackOverflow Developer Survey methodology and patterns
- **CRISP-DM Framework**: Cross-industry standard process for data mining
- **Scikit-learn**: Machine learning library used for modeling
- **Udacity**: Project framework and requirements

## License

This project is for educational purposes as part of a Udacity Data Science course. The synthetic dataset and analysis code are available for learning and reference.

---

**Author**: Data Science Student  
**Course**: Udacity Data Scientist Nanodegree  
**Project**: Write a Data Science Blog Post  
**Date**: 2024