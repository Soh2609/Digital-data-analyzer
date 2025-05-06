
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def clean_data(df):
    """ Basic cleaning: remove duplicates and handle missing values """
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def generate_summary(df):
    if df.empty or df.shape[1] == 0:
        return "⚠️ No columns available to summarize after cleaning. Please check your column selections or dataset."
    return df.describe(include='all')


def generate_correlation_plot(df, save_path='correlation_heatmap.png'):
    """ Create and save a correlation heatmap """
    # Only select numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def auto_generate_report(df, summary, correlation_img_path):
    """ Create a simple HTML report """
    report_html = f"""
    <html>
        <head><title>Data Analysis Report</title></head>
        <body>
            <h1>Automatic Data Analysis Report</h1>
            <h2>Dataset Preview</h2>
            {df.head(10).to_html()}

            <h2>Summary Statistics</h2>
            {summary.to_html()}

            <h2>Correlation Heatmap</h2>
            <img src="{correlation_img_path}" width="700">
        </body>
    </html>
    """
    with open("report.html", "w") as f:
        f.write(report_html)
    return "report.html"
