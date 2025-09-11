import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_diabetes_dataset():
    """
    Comprehensive analysis of the diabetes dataset with visualizations
    """
    print("=== DIABETES DATASET ANALYSIS ===")
    
    # Load the dataset
    df = pd.read_csv('diabetes_dataset.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Total samples: {len(df)}")
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    class_counts = df['is_diabetic'].value_counts()
    print(class_counts)
    print(f"Diabetes prevalence: {class_counts[1]/len(df)*100:.2f}%")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create comprehensive visualizations
    
    # 1. Class Distribution Pie Chart
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    labels = ['Non-Diabetic', 'Diabetic']
    sizes = [class_counts[0], class_counts[1]]
    colors = ['lightblue', 'lightcoral']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Diabetes Class Distribution', fontsize=14, fontweight='bold')
    
    # 2. Feature Distributions
    plt.subplot(2, 2, 2)
    features = ['calories_wk', 'hrs_exercise_wk', 'exercise_intensity', 'annual_income', 'num_children', 'weight']
    
    # BMI-like metric (weight vs exercise)
    df['weight_exercise_ratio'] = df['weight'] / (df['hrs_exercise_wk'] + 0.1)  # Add small constant to avoid division by zero
    
    plt.hist(df[df['is_diabetic']==0]['weight_exercise_ratio'], alpha=0.7, label='Non-Diabetic', bins=50, color='lightblue')
    plt.hist(df[df['is_diabetic']==1]['weight_exercise_ratio'], alpha=0.7, label='Diabetic', bins=50, color='lightcoral')
    plt.xlabel('Weight/Exercise Ratio')
    plt.ylabel('Frequency')
    plt.title('Weight-to-Exercise Ratio Distribution')
    plt.legend()
    
    # 3. Correlation Heatmap
    plt.subplot(2, 2, 3)
    correlation_matrix = df[features + ['is_diabetic']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    
    # 4. Feature Importance (Correlation with Target)
    plt.subplot(2, 2, 4)
    correlations = df[features].corrwith(df['is_diabetic']).abs().sort_values(ascending=True)
    correlations.plot(kind='barh', color='skyblue')
    plt.title('Feature Correlation with Diabetes')
    plt.xlabel('Absolute Correlation')
    
    plt.tight_layout()
    plt.savefig('diabetes_data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed feature analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # Box plot for each feature by diabetes status
        data_non_diabetic = df[df['is_diabetic']==0][feature]
        data_diabetic = df[df['is_diabetic']==1][feature]
        
        ax.boxplot([data_non_diabetic, data_diabetic], 
                   labels=['Non-Diabetic', 'Diabetic'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        
        ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistical test
        stat, p_value = stats.mannwhitneyu(data_non_diabetic, data_diabetic, alternative='two-sided')
        ax.text(0.02, 0.98, f'p-value: {p_value:.2e}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Feature Distributions by Diabetes Status', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Advanced analysis: Feature interactions
    plt.figure(figsize=(15, 10))
    
    # 1. Calories vs Exercise colored by diabetes status
    plt.subplot(2, 3, 1)
    scatter_non_diabetic = plt.scatter(df[df['is_diabetic']==0]['calories_wk'], 
                                     df[df['is_diabetic']==0]['hrs_exercise_wk'], 
                                     alpha=0.6, c='lightblue', label='Non-Diabetic', s=20)
    scatter_diabetic = plt.scatter(df[df['is_diabetic']==1]['calories_wk'], 
                                 df[df['is_diabetic']==1]['hrs_exercise_wk'], 
                                 alpha=0.6, c='lightcoral', label='Diabetic', s=20)
    plt.xlabel('Weekly Calories')
    plt.ylabel('Hours Exercise per Week')
    plt.title('Calories vs Exercise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Weight vs Income
    plt.subplot(2, 3, 2)
    plt.scatter(df[df['is_diabetic']==0]['annual_income'], 
                df[df['is_diabetic']==0]['weight'], 
                alpha=0.6, c='lightblue', label='Non-Diabetic', s=20)
    plt.scatter(df[df['is_diabetic']==1]['annual_income'], 
                df[df['is_diabetic']==1]['weight'], 
                alpha=0.6, c='lightcoral', label='Diabetic', s=20)
    plt.xlabel('Annual Income')
    plt.ylabel('Weight')
    plt.title('Weight vs Income')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Exercise Intensity vs Hours
    plt.subplot(2, 3, 3)
    plt.scatter(df[df['is_diabetic']==0]['hrs_exercise_wk'], 
                df[df['is_diabetic']==0]['exercise_intensity'], 
                alpha=0.6, c='lightblue', label='Non-Diabetic', s=20)
    plt.scatter(df[df['is_diabetic']==1]['hrs_exercise_wk'], 
                df[df['is_diabetic']==1]['exercise_intensity'], 
                alpha=0.6, c='lightcoral', label='Diabetic', s=20)
    plt.xlabel('Hours Exercise per Week')
    plt.ylabel('Exercise Intensity')
    plt.title('Exercise Hours vs Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Age proxy (children) vs Weight
    plt.subplot(2, 3, 4)
    plt.scatter(df[df['is_diabetic']==0]['num_children'], 
                df[df['is_diabetic']==0]['weight'], 
                alpha=0.6, c='lightblue', label='Non-Diabetic', s=20)
    plt.scatter(df[df['is_diabetic']==1]['num_children'], 
                df[df['is_diabetic']==1]['weight'], 
                alpha=0.6, c='lightcoral', label='Diabetic', s=20)
    plt.xlabel('Number of Children')
    plt.ylabel('Weight')
    plt.title('Children vs Weight')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Lifestyle Score (derived feature)
    lifestyle_score = (df['hrs_exercise_wk'] * df['exercise_intensity']) / (df['calories_wk'] / 1000)
    df['lifestyle_score'] = lifestyle_score
    
    plt.subplot(2, 3, 5)
    plt.hist(df[df['is_diabetic']==0]['lifestyle_score'], alpha=0.7, label='Non-Diabetic', bins=50, color='lightblue')
    plt.hist(df[df['is_diabetic']==1]['lifestyle_score'], alpha=0.7, label='Diabetic', bins=50, color='lightcoral')
    plt.xlabel('Lifestyle Score (Exercise*Intensity/Calories)')
    plt.ylabel('Frequency')
    plt.title('Derived Lifestyle Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Risk Factors Summary
    plt.subplot(2, 3, 6)
    
    # Calculate risk factor percentages
    high_calories = df['calories_wk'] > df['calories_wk'].quantile(0.75)
    low_exercise = df['hrs_exercise_wk'] < df['hrs_exercise_wk'].quantile(0.25)
    high_weight = df['weight'] > df['weight'].quantile(0.75)
    
    risk_factors = ['High Calories\n(>75th percentile)', 'Low Exercise\n(<25th percentile)', 'High Weight\n(>75th percentile)']
    
    diabetic_rates = [
        df[high_calories]['is_diabetic'].mean() * 100,
        df[low_exercise]['is_diabetic'].mean() * 100,
        df[high_weight]['is_diabetic'].mean() * 100
    ]
    
    bars = plt.bar(risk_factors, diabetic_rates, color=['red', 'orange', 'darkred'], alpha=0.7)
    plt.ylabel('Diabetes Rate (%)')
    plt.title('Diabetes Rate by Risk Factors')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, diabetic_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_interactions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical Summary
    print("\n=== STATISTICAL ANALYSIS SUMMARY ===")
    
    print("\nFeature Statistics by Diabetes Status:")
    for feature in features:
        non_diabetic = df[df['is_diabetic']==0][feature]
        diabetic = df[df['is_diabetic']==1][feature]
        
        stat, p_value = stats.mannwhitneyu(non_diabetic, diabetic, alternative='two-sided')
        
        print(f"\n{feature.upper()}:")
        print(f"  Non-Diabetic: Mean={non_diabetic.mean():.2f}, Std={non_diabetic.std():.2f}")
        print(f"  Diabetic: Mean={diabetic.mean():.2f}, Std={diabetic.std():.2f}")
        print(f"  Mann-Whitney U test p-value: {p_value:.2e}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
    
    print("\n=== KEY INSIGHTS ===")
    
    # Calculate key insights
    diabetic_group = df[df['is_diabetic']==1]
    non_diabetic_group = df[df['is_diabetic']==0]
    
    print(f"1. Diabetic individuals consume {diabetic_group['calories_wk'].mean():.0f} calories/week vs {non_diabetic_group['calories_wk'].mean():.0f} for non-diabetic")
    print(f"2. Diabetic individuals exercise {diabetic_group['hrs_exercise_wk'].mean():.1f} hours/week vs {non_diabetic_group['hrs_exercise_wk'].mean():.1f} for non-diabetic")
    print(f"3. Average weight: {diabetic_group['weight'].mean():.1f} lbs (diabetic) vs {non_diabetic_group['weight'].mean():.1f} lbs (non-diabetic)")
    print(f"4. Exercise intensity: {diabetic_group['exercise_intensity'].mean():.2f} (diabetic) vs {non_diabetic_group['exercise_intensity'].mean():.2f} (non-diabetic)")
    
    # Risk factor analysis
    high_risk_calories = df['calories_wk'] > 15000
    high_risk_weight = df['weight'] > 250
    low_exercise = df['hrs_exercise_wk'] < 1
    
    print(f"\n5. High calorie intake (>15,000/week) diabetes rate: {df[high_risk_calories]['is_diabetic'].mean()*100:.1f}%")
    print(f"6. High weight (>250 lbs) diabetes rate: {df[high_risk_weight]['is_diabetic'].mean()*100:.1f}%")
    print(f"7. Low exercise (<1 hr/week) diabetes rate: {df[low_exercise]['is_diabetic'].mean()*100:.1f}%")
    
    return df

if __name__ == "__main__":
    df = analyze_diabetes_dataset()
    print("\nData analysis complete! Charts saved as PNG files.")
