from dotenv import load_dotenv
import requests
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import chardet
import sys





# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#  "python-dotenv",
#  "requests",
#  "seaborn",
#  "matplotlib",
#  "chardet",
#  "charset-normalizer"
# ]
# ///

def send_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        detected_encoding = result['encoding']
    print("detected encoding : ",detected_encoding)
    return detected_encoding

def load_token():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the token
    open_api_token = os.getenv("AIPROXY_TOKEN")
    return  open_api_token.strip()

def detect_delimiter_with_pandas(file_path):
    """
    Detect delimiter using Python's csv library by sampling the first row.
    """
    # Read a small sample of the file
    try :
        with open(file_path, "r") as file:
            first_line = file.readline()
    except:
        encoding = send_encoding(file_path)
        with open(file_path, "r",encoding=encoding) as file:
            first_line = file.readline()

    # Check for common delimiters
    delimiters = [",", ";", "|"]
    detected_delimiter = None
    max_splits = 0

    for delimiter in delimiters:
        splits = len(first_line.split(delimiter))
        if splits > max_splits:
            max_splits = splits
            detected_delimiter = delimiter

    return detected_delimiter

def process_csv_with_pandas(file_path, delimiter=None):
    """
    Process CSV file with Pandas after detecting the delimiter.
    """
    # Detect delimiter if not provided
    if delimiter is None:
        delimiter = detect_delimiter_with_pandas(file_path)

    # Load CSV with Pandas
    df = pd.read_csv(file_path, delimiter=delimiter,encoding=send_encoding(file_path))
    print(f"Detected delimiter: {delimiter}")
    return df

def clean_columns(column_list, df_columns):
    """
    Clean and validate column names.
    """
    cleaned_columns = []
    for col in column_list:
        # Remove backticks and split concatenated columns
        cols = [c.strip("`'\" ") for c in col.replace("`", "").split(",") if c.strip()]
        for c in cols:
            if c in df_columns:  # Check if the column exists in the DataFrame
                cleaned_columns.append(c)
            else:
                print(f"Warning: Column '{c}' not found in DataFrame.")
    return cleaned_columns

def visualize_eda_and_save(df, parsed_response,file_path):
    """
    Generate visualizations based on parsed GPT response and save them as images.
    """
    location_of_images=[]
    cwd=os.getcwd()
    # Ensure the output directory exists
    output_dir = f"{cwd}"
    print(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    print(os.path.join(output_dir, "correlation_heatmap.png"))

    df_columns = df.columns.tolist()
    parsed_response = {key: clean_columns(value, df_columns) for key, value in parsed_response.items()}
    print("post cleaning")
    print(parsed_response)
    # 1. Pearson Correlation Heatmap
    pearson_columns = parsed_response.get("Pearson_correlation", [])
    if len(pearson_columns) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[pearson_columns].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Pearson Correlation Heatmap")
        plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
        location_of_images.append(os.path.join(output_dir, "correlation_heatmap.png"))
        plt.close()

    # 2. Boxplot for Outliers
    boxplot_columns = parsed_response.get("Boxplot", [])
    if len(boxplot_columns) == 2:
        categorical_col, numerical_col = boxplot_columns
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=categorical_col, y=numerical_col, data=df)
        plt.title(f"Boxplot of {numerical_col} by {categorical_col}")
        plt.xticks(rotation=45)
        location_of_images.append(os.path.join(output_dir, "boxplot_outliers.png"))
        plt.savefig(os.path.join(output_dir, "boxplot_outliers.png"))
        plt.close()

    # 3. Pairplot for Numerical Columns
    pairplot_columns = parsed_response.get("Pairplot", [])
    if len(pairplot_columns) > 1:
        pairplot = sns.pairplot(df[pairplot_columns])
        pairplot.fig.suptitle("Pairplot of Numerical Columns", y=1.02)
        pairplot.savefig(os.path.join(output_dir, "pairplot.png"))
        location_of_images.append(os.path.join(output_dir, "pairplot.png"))
        plt.close()

    # 4. Histogram for Numerical Columns
    histogram_columns = parsed_response.get("Histogram", [])
    for col in histogram_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=20)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"histogram_{col}.png"))
        location_of_images.append(os.path.join(output_dir, f"histogram_{col}.png"))
        plt.close()
    return  location_of_images

def parse_gpt_response(gpt_output):
    """
    Parse GPT response into a structured dictionary for EDA visualizations.
    Handles formatted GPT response with headings and explanations.
    """

    parsed_response = {
        "Pearson_correlation": [],
        "Boxplot": [],
        "Pairplot": [],
        "Histogram": []
    }

    lines = gpt_output.splitlines()
    print("linesparsing debugging")
    print(lines)


    current_section = None
    is_parsing_columns = False  # Flag to indicate we're in a "Relevant Columns" section

    for line in lines:
        line = line.strip()  # Remove leading/trailing spaces

        # Identify the section based on headings
        if line.startswith("### 1. Pearson Correlation Heatmap"):
            current_section = "Pearson_correlation"
            is_parsing_columns = False
        elif line.startswith("### 2. Boxplot"):
            current_section = "Boxplot"
            is_parsing_columns = False
        elif line.startswith("### 3. Pairplot"):
            current_section = "Pairplot"
            is_parsing_columns = False
        elif line.startswith("### 4. Histogram"):
            current_section = "Histogram"
            is_parsing_columns = False

        # Start collecting columns when specific headers appear
        elif current_section and ("**Relevant Columns:**" in line or "**Columns to Use:**" in line):
            is_parsing_columns = True

        # Collect bullet-pointed columns
        elif is_parsing_columns and line.startswith("-"):
            # Clean up the column name
            col = line.lstrip("- ").strip("`'\"").split("(")[0].strip()  # Remove bullet, backticks, quotes, and parentheses
            if col:
                # Handle concatenated columns (e.g., `col1`, `col2`)
                parsed_response[current_section].extend([c.strip() for c in col.split(",") if c.strip()])

        # Stop parsing columns when a non-bullet-point line is encountered
        elif is_parsing_columns and not line.startswith("-"):
            is_parsing_columns = False

    print(parsed_response)
    return parsed_response

def analyze_dataset_with_gpt_and_save(df, api_key,file_path):
    """
    Use GPT to suggest columns for EDA visualizations, parse the response, and generate visualizations.
    """
    # Extract column names and sample data
    columns = df.columns.tolist()
    sample_data = df.head(3).to_dict(orient="records")
    location_of_images=[]
    print("Sample Data Sent to GPT:")
    print(sample_data)

    # Prompt for GPT
    old_prompt = (
        "You are assisting in exploratory data analysis (EDA) for a dataset. "
        "Given the following dataset columns and sample data, "
        "suggest a few columns that would be most relevant for creating visualizations. "
        "Explain why these columns are suitable for each visualization. "
        "The visualizations include: \n"
        "1. Pearson correlation heatmap (relationships between numerical columns)\n"
        "2. Boxplot (to check for outliers in numerical columns across categories)\n"
        "3. Pairplot (pairwise relationships between numerical columns)\n"
        "4. Histogram (to understand the distribution of numerical columns).\n\n"
        f"Columns: {columns}\n\nSample Data:\n{sample_data}"
    )

    prompt = (
        "You are assisting in exploratory data analysis (EDA) for a dataset. "
        "Given the following dataset columns and sample data, "
        "suggest all relevant columns that would be most suitable for each type of visualization. "
        "For each visualization type, list the relevant columns under a structured heading, followed by an explanation for their suitability. "
        "Ensure the response follows this exact structure and includes all relevant columns without any limit:\n\n"
        "### 1. Pearson Correlation Heatmap\n"
        "**Relevant Columns:**\n"
        "- `column_name1`\n"
        "- `column_name2`\n"
        "- `column_nameN` (continue listing all relevant numerical columns)\n"
        "**Reason:** Explanation of why these columns are suitable for a Pearson correlation heatmap.\n\n"
        "### 2. Boxplot\n"
        "**Relevant Columns:**\n"
        "- `numerical_column_name`\n"
        "- `categorical_column_name`\n"
        "**Reason:** Explanation of why these columns are suitable for a boxplot.\n\n"
        "### 3. Pairplot\n"
        "**Relevant Columns:**\n"
        "- `column_name1`\n"
        "- `column_name2`\n"
        "- `column_nameN` (list all numerical columns that should be pairwise compared)\n"
        "**Reason:** Explanation of why these columns are suitable for a pairplot.\n\n"
        "### 4. Histogram\n"
        "**Relevant Columns:**\n"
        "- `column_name1`\n"
        "- `column_nameN` (list all numerical columns that should have histograms)\n"
        "**Reason:** Explanation of why these columns are suitable for histograms.\n\n"
        f"Columns: {columns}\n\nSample Data:\n{sample_data}"
    )

    # API call to GPT
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, headers=headers, json=data,verify=False)

    # Process GPT response
    if response.status_code == 200:
        result = response.json()
        print("GPT Response:")
        gpt_output = result['choices'][0]['message']['content']
        print(gpt_output)

        # Parse GPT response into structured format
        parsed_response = parse_gpt_response(gpt_output)

        # Generate and save visualizations
        location_of_images=visualize_eda_and_save(df, parsed_response,file_path)
        print(f"Visualizations saved in './file_path'")

    else:
        print(f"Error: {response.status_code}, {response.text}")
    return  location_of_images

def generate_dataset_stats(df):
    """
    Generate standard statistics for a dataset and ensure all values are JSON serializable.
    """
    stats = {}

    for column in df.columns:
        col_data = df[column]
        col_stats = {
            "Column Name": column,
            "Data Type": col_data.dtype.name,
            "Unique Values": int(col_data.nunique()),
            "Missing Values": int(col_data.isnull().sum()),
            "Non-Null Count": int(col_data.notnull().sum())
        }

        # Stats for numerical columns
        if pd.api.types.is_numeric_dtype(col_data):
            col_stats.update({
                "Mean": float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                "Median": float(col_data.median()) if not pd.isna(col_data.median()) else None,
                "Std Dev": float(col_data.std()) if not pd.isna(col_data.std()) else None,
                "Min": float(col_data.min()) if not pd.isna(col_data.min()) else None,
                "Max": float(col_data.max()) if not pd.isna(col_data.max()) else None,
                "25%": float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else None,
                "50%": float(col_data.quantile(0.50)) if not pd.isna(col_data.quantile(0.50)) else None,
                "75%": float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else None,
            })

        # Stats for categorical columns
        elif pd.api.types.is_object_dtype(col_data):
            mode = col_data.mode()
            col_stats.update({
                "Mode": str(mode[0]) if not mode.empty else None,
                "Mode Frequency": int(col_data.value_counts().iloc[0]) if not col_data.value_counts().empty else None,
                "Top Categories": {str(k): int(v) for k, v in col_data.value_counts().head(5).items()}
            })

        # Stats for datetime columns
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_stats.update({
                "Earliest Date": str(col_data.min()),
                "Latest Date": str(col_data.max()),
                "Unique Dates": int(col_data.nunique())
            })

        # Stats for boolean columns
        elif pd.api.types.is_bool_dtype(col_data):
            col_stats.update({
                "True Count": int(col_data.sum()),
                "False Count": int(len(col_data) - col_data.sum())
            })

        stats[column] = col_stats

    return stats


def send_stats_to_gpt(stats):
    """
    Send dataset statistics to GPT-4o-mini for generating a business-oriented summary.
    """
    # Prepare the prompt
    prompt = (
            "You are a business data analyst. Based on the provided dataset statistics, "
            "create a meaningful summary for a business audience. Focus on:\n"
            "- Key trends in the dataset.\n"
            "- Significant patterns or outliers and their potential business implications.\n"
            "- Dominant categories or groups and what they suggest about the dataset.\n"
            "- Issues with missing data and how they might affect business decisions.\n"
            "- Any other insights that can help stakeholders better understand this dataset.\n\n"
            "Here are the dataset statistics:\n" + json.dumps(stats, indent=2)
    )

    # GPT API Endpoint
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

    # Request Headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {load_token()}"
    }

    # GPT API Payload
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    # Send the HTTP request
    response = requests.post(url, headers=headers, json=data, verify=False)

    # Handle response
    if response.status_code == 200:
        result = response.json()
        summary = result['choices'][0]['message']['content']
        return summary
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")

def build_readme(summary, chart_paths, repo_dir="."):
    """
    Build or update a README.md file with a summary and EDA visualizations.

    Args:
        summary (str): The business-oriented summary to include in the README.
        chart_paths (list): List of file paths to charts for EDA visuals.
        repo_dir (str): Path to the repository directory where README.md resides. Defaults to the current directory.
    """
    # Define the README file path
    readme_path = os.path.join(repo_dir, "Readme.md")

    # Check if README already exists and load its content if it does
    if os.path.exists(readme_path):
        with open(readme_path, "r") as readme_file:
            readme_content = readme_file.read()
    else:
        # Start with a new template if no README exists
        readme_content = "# Project Overview\n\n"

    # Add or update the EDA Summary section
    if "## EDA Summary" in readme_content:
        readme_content = readme_content.split("## EDA Summary")[0]

    readme_content += "## EDA Summary\n\n"
    readme_content += summary + "\n\n"

    # Add or update the EDA Visuals section
    readme_content += "## EDA Visuals\n\n"
    for chart_path in chart_paths:
        chart_filename = os.path.basename(chart_path)
        relative_path = os.path.relpath(chart_path, repo_dir)
        readme_content += f"![{chart_filename}]({relative_path})\n\n"

    # Save the updated README.md file
    with open(readme_path, "w") as readme_file:
        readme_file.write(readme_content)

    print(f"README.md updated successfully at {readme_path}")





if len(sys.argv) != 2:
    print("Usage: uv run autolysis.py dataset.csv")
    sys.exit(1)

file_path_src = sys.argv[1]
# file_path_src = "goodreads.csv"  # Replace with the path to your CSV file
df_raw = process_csv_with_pandas(file_path_src)
print(df_raw.head(5))
location_of_images=analyze_dataset_with_gpt_and_save(df_raw, load_token(),file_path_src.split(".")[0])
print("Image location")
dataset_stats = generate_dataset_stats(df_raw)
summary=""" There is no summar for now its on the way"""
try:
    summary = send_stats_to_gpt(dataset_stats)
    print("Business-Oriented Dataset Summary:")
    print(summary)
except Exception as e:
    print(e)
print("Summary here")
print(summary)
print("location of images")
print(location_of_images)
print("get working dir")
print(os.getcwd())
build_readme(summary, location_of_images,os.path.join(os.getcwd()))
#
