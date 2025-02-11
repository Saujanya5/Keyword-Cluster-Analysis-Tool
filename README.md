# Keyword Cluster Analysis Tool

## Overview
This tool performs advanced clustering analysis on keyword data from digital marketing campaigns, helping marketers optimize their ad groups and campaign structure. It uses natural language processing and hierarchical clustering to group similar keywords based on semantic meaning while considering their performance metrics.

## Business Impact
| Benefit | Description |
|---------|------------|
| **Optimize Ad Group Structure** | Automatically identify keyword patterns and suggest optimized ad group structures, leading to better Quality Scores and reduced CPCs. |
| **Performance Analysis** | Gain insights into keyword cluster performance through metrics like CTR, conversion rates, and overall performance scores. |
| **Campaign Efficiency** | Save hours of manual keyword analysis and grouping work. |
| **Data-Driven Decisions** | Make informed decisions about campaign restructuring based on actual performance data. |
| **Scale Management** | Efficiently handle large keyword sets with automated clustering. |

## Features
- **Semantic keyword clustering** using TF-IDF vectorization.
- **Hierarchical clustering** with customizable cluster sizes.
- **Performance metrics calculation** (CTR, conversion rates).
- **Automated ad group suggestions**.
- **Comprehensive performance scoring**.
- **Detailed cluster analysis and reporting**.

## Requirements
```bash
python
pandas
numpy
scikit-learn
nltk
```

## Input Format
The tool expects a CSV file with the following columns:

| Column Name | Description |
|-------------|-------------|
| `keyword` | The search term or keyword |
| `clicks` | Number of clicks received |
| `impressions` | Number of impressions |
| `conversions` | Number of conversions |
| `ad_group` | Current ad group name |

### Example Input Data

| keyword | clicks | impressions | conversions | ad_group |
|---------|--------|------------|-------------|-----------|
| buy blue shoes | 100 | 1000 | 10 | Shoes_General |
| purchase blue sneakers | 80 | 900 | 8 | Shoes_General |
| red running shoes | 150 | 1500 | 15 | Running_Shoes |

## Output
The tool generates two types of output:

### 1. Console Output
- Top keywords per cluster.
- Performance metrics (CTR, conversion rates).
- Cluster sizes.
- Suggested ad group names.

### 2. CSV Output (`keyword_cluster_results.csv`)
Contains detailed clustering results with the following columns:

| Column Name | Description |
|-------------|-------------|
| `cluster` | Cluster identifier |
| `top_keywords` | Most common keywords in the cluster |
| `keyword_count` | Number of keywords in the cluster |
| `clicks` | Total clicks in the cluster |
| `impressions` | Total impressions in the cluster |
| `conversions` | Total conversions in the cluster |
| `ctr` | Click-through rate (%) |
| `conversion_rate` | Conversion rate (%) |
| `performance_score` | Performance score based on CTR and conversion rate |
| `ad_group` | Original ad group |
| `keyword` | Keyword in the cluster |

### Example Console Output
```
Cluster 0:
Top Keywords: blue, shoes, sneakers
Number of Keywords: 25
Clicks: 500
Impressions: 5000
CTR: 10%
Conversion Rate: 8.5%
Performance Score: 0.85
Main Ad Group: Blue_Shoes
Suggested Ad Group Name: Cluster_0_Blue_Shoes
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/keyword-cluster-tool.git](https://github.com/Saujanya5/Keyword-Cluster-Analysis-Tool
   cd keyword-cluster-tool
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python keyword-cluster.py
   ```
4. Enter the path to your CSV file when prompted.
5. Review the console output and check the generated CSV file for detailed results.

## Configuration
You can customize the clustering behavior by modifying these parameters:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `max_cluster_size` | Maximum keywords per cluster | 50 |
| `min_cluster_size` | Minimum keywords per cluster | 15 |


