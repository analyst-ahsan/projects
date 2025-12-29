"""
EDA Plotting Utilities Analysis
Author: Data Science Portfolio Project
Description: Simple plotting functions for univariate and bivariate visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def identify_column_types(df, target_col=None):
    """
    Automatically identify numerical and categorical columns
    
    Parameters:
    -----------
    df : pandas DataFrame
        Input dataframe
    target_col : str, optional
        Target column name to exclude from features
    
    Returns:
    --------
    dict : Dictionary with 'numerical' and 'categorical' column lists
    """
    # Exclude target if specified
    cols = [col for col in df.columns if col != target_col]
    
    numerical_cols = []
    categorical_cols = []
    
    for col in cols:
        # Check if column is numerical
        if df[col].dtype in ['int64', 'float64']:
            # If unique values < 10, might be categorical
            if df[col].nunique() < 10:
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    print(f"Found {len(numerical_cols)} numerical columns")
    print(f"Found {len(categorical_cols)} categorical columns")
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


# ============================================================================
# UNIVARIATE ANALYSIS
# ============================================================================

def plot_numerical_univariate(df, numerical_cols, figsize=(15, 5), save_path=None):
    """
    Plot distributions for numerical columns
    
    Parameters:
    -----------
    df : pandas DataFrame
    numerical_cols : list
        List of numerical column names
    figsize : tuple
        Figure size for each subplot
    save_path : str, optional
        Path to save plots (e.g., 'plots/numerical_univariate')
    """
    for col in numerical_cols:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'Distribution: {col}', fontsize=16, fontweight='bold')
        
        # 1. Histogram
        axes[0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Histogram')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[0].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        axes[0].legend()
        
        # 2. Box plot
        axes[1].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='black'),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black'),
                       capprops=dict(color='black'))
        axes[1].set_ylabel(col)
        axes[1].set_title('Box Plot')
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. KDE (Density plot)
        df[col].dropna().plot(kind='density', ax=axes[2], color='darkblue', linewidth=2)
        axes[2].set_xlabel(col)
        axes[2].set_ylabel('Density')
        axes[2].set_title('Kernel Density Plot')
        axes[2].grid(alpha=0.3)
        axes[2].fill_between(axes[2].lines[0].get_xdata(), axes[2].lines[0].get_ydata(), alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{col}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print basic info
        print(f"\n{col}:")
        print(f"  Mean: {mean_val:.2f}, Median: {median_val:.2f}")
        print(f"  Min: {df[col].min():.2f}, Max: {df[col].max():.2f}")
        print(f"  Missing: {df[col].isna().sum()} ({df[col].isna().mean()*100:.2f}%)\n")


def plot_categorical_univariate(df, categorical_cols, figsize=(12, 5), save_path=None):
    """
    Plot distributions for categorical columns
    
    Parameters:
    -----------
    df : pandas DataFrame
    categorical_cols : list
        List of categorical column names
    figsize : tuple
        Figure size for each subplot
    save_path : str, optional
        Path to save plots
    """
    for col in categorical_cols:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Distribution: {col}', fontsize=16, fontweight='bold')
        
        # Get value counts
        value_counts = df[col].value_counts()
        
        # 1. Bar plot
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[0], palette='viridis')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Count Plot')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for i, v in enumerate(value_counts.values):
            axes[0].text(i, v + max(value_counts.values)*0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Pie chart
        colors = sns.color_palette('viridis', len(value_counts))
        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('Percentage Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{col}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print basic info
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing: {df[col].isna().sum()} ({df[col].isna().mean()*100:.2f}%)\n")


def univariate_analysis(df, target_col=None, save_path=None):
    """
    Complete univariate analysis - automatically detects and plots all columns
    
    Parameters:
    -----------
    df : pandas DataFrame
    target_col : str, optional
        Target column to exclude from analysis
    save_path : str, optional
        Directory path to save plots
    
    Usage:
    ------
    univariate_analysis(df, target_col='Churn')
    univariate_analysis(df, target_col='Churn', save_path='plots/univariate')
    """
    print("="*80)
    print("UNIVARIATE ANALYSIS")
    print("="*80)
    
    # Identify column types
    col_types = identify_column_types(df, target_col)
    
    # Plot numerical columns
    if col_types['numerical']:
        print(f"\n\nPlotting {len(col_types['numerical'])} Numerical Columns...")
        print("-"*80)
        plot_numerical_univariate(df, col_types['numerical'], 
                                 save_path=f"{save_path}/numerical" if save_path else None)
    
    # Plot categorical columns
    if col_types['categorical']:
        print(f"\n\nPlotting {len(col_types['categorical'])} Categorical Columns...")
        print("-"*80)
        plot_categorical_univariate(df, col_types['categorical'],
                                   save_path=f"{save_path}/categorical" if save_path else None)


# ============================================================================
# BIVARIATE ANALYSIS
# ============================================================================

def plot_numerical_vs_target(df, numerical_cols, target_col, figsize=(15, 5), save_path=None):
    """
    Plot numerical features vs target variable
    
    Parameters:
    -----------
    df : pandas DataFrame
    numerical_cols : list
        List of numerical column names
    target_col : str
        Target column name (should be binary: 0/1)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save plots
    """
    target_labels = sorted(df[target_col].unique())
    
    for col in numerical_cols:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'{col} vs {target_col}', fontsize=16, fontweight='bold')
        
        # 1. Overlapping Histograms
        for label in target_labels:
            subset = df[df[target_col] == label][col].dropna()
            axes[0].hist(subset, alpha=0.6, label=f'{target_col}={label}', bins=30, edgecolor='black')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution by Target')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. Box plot by target
        df_plot = df[[col, target_col]].dropna()
        df_plot[target_col] = df_plot[target_col].astype(str)
        sns.boxplot(data=df_plot, x=target_col, y=col, ax=axes[1], palette='Set2')
        axes[1].set_title('Box Plot by Target')
        axes[1].set_xlabel(target_col)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. Violin plot
        sns.violinplot(data=df_plot, x=target_col, y=col, ax=axes[2], palette='Set2')
        axes[2].set_title('Violin Plot by Target')
        axes[2].set_xlabel(target_col)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_num_{col}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print mean comparison
        print(f"\n{col} by {target_col}:")
        for label in target_labels:
            mean_val = df[df[target_col] == label][col].mean()
            print(f"  {target_col}={label}: Mean = {mean_val:.2f}")
        print()


def plot_categorical_vs_target(df, categorical_cols, target_col, figsize=(14, 5), save_path=None):
    """
    Plot categorical features vs target variable
    
    Parameters:
    -----------
    df : pandas DataFrame
    categorical_cols : list
        List of categorical column names
    target_col : str
        Target column name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save plots
    """
    for col in categorical_cols:
        # Create crosstab
        ct = pd.crosstab(df[col], df[target_col], normalize='index') * 100
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'{col} vs {target_col}', fontsize=16, fontweight='bold')
        
        # 1. Grouped bar chart (percentage)
        ct.plot(kind='bar', stacked=False, ax=axes[0], color=['#2ecc71', '#e74c3c'], 
                edgecolor='black', alpha=0.8)
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Percentage (%)')
        axes[0].set_title('Target Rate by Category')
        axes[0].legend(title=target_col, labels=[f'{target_col}=0', f'{target_col}=1'])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.1f%%', label_type='edge')
        
        # 2. Count plot with hue
        df_plot = df[[col, target_col]].copy()
        df_plot[target_col] = df_plot[target_col].astype(str)
        sns.countplot(data=df_plot, x=col, hue=target_col, ax=axes[1], palette='Set2')
        axes[1].set_xlabel(col)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Count by Category and Target')
        axes[1].legend(title=target_col, labels=[f'{target_col}=0', f'{target_col}=1'])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_cat_{col}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print churn rate by category
        print(f"\n{target_col}=1 Rate by {col}:")
        print(ct[1].sort_values(ascending=False).to_string())
        print()


def plot_correlation_matrix(df, numerical_cols, figsize=(12, 10), save_path=None):
    """
    Plot correlation heatmap for numerical features
    
    Parameters:
    -----------
    df : pandas DataFrame
    numerical_cols : list
        List of numerical column names
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save plot
    """
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_correlation_matrix.png", dpi=300, bbox_inches='tight')
    
    plt.show()


def bivariate_analysis(df, target_col, include_correlation=True, save_path=None):
    """
    Complete bivariate analysis - plots all features vs target
    
    Parameters:
    -----------
    df : pandas DataFrame
    target_col : str
        Target column name
    include_correlation : bool
        Whether to include correlation matrix
    save_path : str, optional
        Directory path to save plots
    
    Usage:
    ------
    bivariate_analysis(df, target_col='Churn')
    bivariate_analysis(df, target_col='Churn', save_path='plots/bivariate')
    bivariate_analysis(df, target_col='Churn', include_correlation=False)
    """
    print("="*80)
    print("BIVARIATE ANALYSIS")
    print("="*80)
    
    # Identify column types
    col_types = identify_column_types(df, target_col)
    
    # Numerical vs Target
    if col_types['numerical']:
        print(f"\n\nPlotting {len(col_types['numerical'])} Numerical Features vs Target...")
        print("-"*80)
        plot_numerical_vs_target(df, col_types['numerical'], target_col,
                                save_path=f"{save_path}/bivariate" if save_path else None)
    
    # Categorical vs Target
    if col_types['categorical']:
        print(f"\n\nPlotting {len(col_types['categorical'])} Categorical Features vs Target...")
        print("-"*80)
        plot_categorical_vs_target(df, col_types['categorical'], target_col,
                                  save_path=f"{save_path}/bivariate" if save_path else None)
    
    # Correlation matrix
    if include_correlation and col_types['numerical']:
        print(f"\n\nPlotting Correlation Matrix...")
        print("-"*80)
        # Add target to correlation if it's numerical
        if df[target_col].dtype in ['int64', 'float64']:
            corr_cols = col_types['numerical'] + [target_col]
        else:
            corr_cols = col_types['numerical']
        
        plot_correlation_matrix(df, corr_cols, 
                               save_path=f"{save_path}/bivariate" if save_path else None)


# ============================================================================
# QUICK USAGE EXAMPLES
# ============================================================================

"""
USAGE EXAMPLES:

# Import in your notebook
from eda_utils import univariate_analysis, bivariate_analysis

# 1. Univariate Analysis (describe each variable)
univariate_analysis(df, target_col='Churn')

# Save plots
univariate_analysis(df, target_col='Churn', save_path='plots/univariate')


# 2. Bivariate Analysis (each variable vs target)
bivariate_analysis(df, target_col='Churn')

# Without correlation matrix
bivariate_analysis(df, target_col='Churn', include_correlation=False)

# Save plots
bivariate_analysis(df, target_col='Churn', save_path='plots/bivariate')


# 3. Individual functions for specific columns
from eda_utils import plot_numerical_univariate, plot_categorical_univariate
from eda_utils import plot_numerical_vs_target, plot_categorical_vs_target

# Plot specific numerical columns
plot_numerical_univariate(df, ['tenure', 'MonthlyCharges', 'TotalCharges'])

# Plot specific categorical columns
plot_categorical_univariate(df, ['Contract', 'PaymentMethod'])

# Plot specific features vs target
plot_numerical_vs_target(df, ['tenure', 'MonthlyCharges'], target_col='Churn')
plot_categorical_vs_target(df, ['Contract', 'InternetService'], target_col='Churn')
"""