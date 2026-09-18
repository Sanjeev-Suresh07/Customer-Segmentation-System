# Data-Driven Customer Segmentation and Behavior Analysis for Personalized Marketing

## Project Overview

This project analyzes e-commerce customer behavior and uses unsupervised machine learning to identify meaningful customer segments.

The system applies K-Means clustering to customer demographic, purchasing, membership, and discount-related characteristics. The resulting customer segments are analyzed to understand behavioral patterns and derive targeted marketing strategies.

An interactive Streamlit dashboard is provided to explore customer segments, model performance, customer-level records, and marketing insights.

---

## Objectives

- Analyze customer purchasing behavior.
- Perform data understanding, cleaning, and exploratory data analysis.
- Identify meaningful customer segments using unsupervised learning.
- Apply appropriate feature preprocessing and scaling techniques.
- Determine the number of clusters using clustering evaluation techniques.
- Evaluate the final clustering model using the Silhouette Score.
- Profile the identified customer segments.
- Generate personalized marketing strategies for each segment.
- Develop an interactive dashboard for customer segmentation analysis.

---

## Dataset

### E-Commerce Customer Behavior Dataset

The project uses an e-commerce customer behavior dataset containing customer demographic and purchasing information.

### Dataset Details

- Number of customers: 350
- Number of original columns: 11
- Source: Kaggle
- Dataset type: Customer behavior data

### Main Attributes

- Customer ID
- Gender
- Age
- City
- Membership Type
- Total Spend
- Items Purchased
- Average Rating
- Discount Applied
- Days Since Last Purchase
- Satisfaction Level

---

## Methodology

The complete project workflow is:

```text
Dataset Collection
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Feature Preprocessing
        ↓
K-Means Clustering
        ↓
Cluster Evaluation
        ↓
Customer Segmentation
        ↓
Segment Profiling
        ↓
Marketing Insights
        ↓
Interactive Dashboard

Machine Learning Approach
Learning Type

Unsupervised Learning

Algorithm

K-Means Clustering

K-Means clustering is used to group customers according to similarities in their characteristics without requiring predefined customer labels.

The algorithm assigns customers to clusters based on their distance from cluster centroids and iteratively updates the centroids until the clustering stabilizes.

Final Model Configuration
Algorithm: K-Means Clustering
Number of clusters: 7
Random state: 42
n_init: 20
Numerical Features

The final model uses the following numerical features:

Age
Total Spend
Categorical Features

The final model uses the following categorical features:

Membership Type
Discount Applied
Feature Preprocessing

Numerical features are scaled using:

MinMaxScaler

Categorical features are transformed using:

OneHotEncoder

The processed numerical and categorical features are combined to create the final feature space used by the K-Means clustering algorithm.

Model Evaluation

Since the project uses unsupervised learning, there are no predefined target labels for calculating classification accuracy.

Therefore, the Silhouette Score is used to evaluate the quality of the clustering.

Final Evaluation Result
Number of Clusters: 7
Silhouette Score: 0.948270

The Silhouette Score measures how well each customer fits within its assigned cluster compared with other clusters.

A higher score indicates stronger separation between clusters and greater similarity among customers within the same cluster.

The final model achieved a Silhouette Score of 0.948270.

Final Customer Segments

The final K-Means model identifies seven customer segments based on their demographic, purchasing, membership, and discount characteristics.

Customer Segment	Customers
Bronze Discount Customers	58
Bronze Regular Customers	58
Gold High Value Customers	59
Gold Premium Customers	58
Silver Inactive Customers	34
Silver Moderate Customers	24
Silver Regular Customers	59
Customer Segment Profiles
Bronze Discount Customers
Customers: 58
Average Spend: ₹499.88
Average Items Purchased: 9.41
Average Rating: 3.46
Average Days Since Purchase: 40.47

Marketing Strategy:
Use targeted discounts, bundles, and personalized offers to increase purchase frequency.

Bronze Regular Customers
Customers: 58
Average Spend: ₹446.89
Average Items Purchased: 7.57
Average Rating: 3.19
Average Days Since Purchase: 22.76

Marketing Strategy:
Encourage repeat purchases through loyalty rewards and personalized product recommendations.

Gold High Value Customers
Customers: 59
Average Spend: ₹1165.04
Average Items Purchased: 15.27
Average Rating: 4.54
Average Days Since Purchase: 24.59

Marketing Strategy:
Focus on customer retention, loyalty programs, and personalized offers.

Gold Premium Customers
Customers: 58
Average Spend: ₹1459.77
Average Items Purchased: 20.00
Average Rating: 4.81
Average Days Since Purchase: 11.17

Marketing Strategy:
Provide exclusive offers, premium products, and VIP benefits.

Silver Inactive Customers
Customers: 34
Average Spend: ₹703.69
Average Items Purchased: 12.76
Average Rating: 4.02
Average Days Since Purchase: 53.18

Marketing Strategy:
Use re-engagement campaigns, reminders, and limited-time offers.

Silver Moderate Customers
Customers: 24
Average Spend: ₹671.55
Average Items Purchased: 10.04
Average Rating: 3.80
Average Days Since Purchase: 34.62

Marketing Strategy:
Encourage repeat purchases through bundles and personalized promotions.

Silver Regular Customers
Customers: 59
Average Spend: ₹805.49
Average Items Purchased: 11.68
Average Rating: 4.17
Average Days Since Purchase: 15.27

Marketing Strategy:
Use loyalty rewards and personalized product recommendations.

Exploratory Data Analysis

The project includes exploratory analysis of customer behavior using:

Dataset dimensions and structure
Data types
Missing value analysis
Duplicate record analysis
Statistical summaries
Gender distribution
Membership type distribution
Satisfaction level distribution
Age distribution
Total spending distribution
Outlier analysis
Correlation analysis
Pairwise relationships between numerical features

Visualizations were created using Matplotlib and Seaborn.

Interactive Dashboard

The project includes an interactive Streamlit dashboard for exploring the final customer segmentation model.

Dashboard Features
Customer segmentation overview
Interactive segment explorer
Segment-level statistics
Customer distribution visualization
Spending vs. items purchased visualization
Interactive membership filters
Interactive discount filters
Segment summary table
Customer-level data explorer
Customer ID search
Filtered customer data download
Model configuration display
Silhouette Score display
Marketing strategy recommendations
Dashboard Sections
1. Segment Explorer

Allows users to select a customer segment and view:

Number of customers
Average spending
Average items purchased
Days since last purchase
Segment description
Recommended marketing strategy
2. Segment Summary

Provides a comparative view of all seven customer segments and their major behavioral characteristics.

3. Customer Data

Displays customer-level records along with their assigned cluster and customer segment.

Users can search for customer IDs and download the filtered customer data.

System Architecture

                    Customer Dataset
                           |
                           ↓
                  Data Understanding
                           |
                           ↓
                     Data Cleaning
                           |
                           ↓
                 Exploratory Data Analysis
                           |
                           ↓
                    Feature Selection
                           |
                           ↓
                 Feature Preprocessing
                  /                  \
                 ↓                    ↓
          MinMaxScaler          OneHotEncoder
                 \                    /
                  ↓                  ↓
                   Combined Features
                           |
                           ↓
                   K-Means Clustering
                           |
                           ↓
                  Cluster Evaluation
                           |
                           ↓
                  Customer Segments
                           |
                           ↓
                  Segment Profiling
                           |
                           ↓
                 Marketing Insights
                           |
                           ↓
                 Streamlit Dashboard

Technologies Used
Programming Language
Python
Data Analysis
Pandas
NumPy
Data Visualization
Matplotlib
Seaborn
Plotly
Machine Learning
Scikit-learn
K-Means Clustering
MinMaxScaler
OneHotEncoder
Silhouette Score
Dashboard
Streamlit
Model Persistence
Joblib
Development Tools
Jupyter Notebook
Visual Studio Code
Git
GitHub


Project Structure
Customer-Segmentation-System/
│
├── data/
│   ├── customer_behavior.csv
│   ├── cleaned_customer_behavior.csv
│   └── customer_segments.csv
│
├── models/
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── kmeans_model.pkl
│
├── notebooks/
│   ├── 01_Data_Understanding_and_Cleaning.ipynb
│   └── 02_Customer_Segmentation.ipynb
│
├── src/
│   ├── clustering.py
│   ├── evaluation.py
│   └── business_insights.py
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore

How to Run the Project
1. Clone the Repository
git clone https://github.com/Sanjeev-Suresh07/Customer-Segmentation-System.git
cd Customer-Segmentation-System

2. Create a Virtual Environment
python -m venv .venv

3. Activate the Virtual Environment

For Windows:

.venv\Scripts\activate

4. Install Dependencies
pip install -r requirements.txt

5. Run the Streamlit Dashboard
streamlit run app.py

The application will open in the browser using the local Streamlit server.

Model Output

The final system produces:

Seven customer segments
Customer-level cluster assignments
Segment-level behavioral profiles
Marketing recommendations
Interactive visualizations
Model evaluation metrics
Filterable customer records
Downloadable customer data
Business Insights

The segmentation model provides a data-driven way to understand different customer groups.

The identified segments demonstrate differences in:

Customer spending
Purchase quantity
Membership type
Discount usage
Customer engagement
Purchase recency
Customer ratings

These differences can be used to design targeted marketing strategies rather than applying the same campaign to all customers.

Limitations
The dataset contains 350 customer records.
The clustering results depend on the selected features and preprocessing methods.
K-Means assumes that clusters can be represented effectively using centroid-based grouping.
The marketing strategies are analytical recommendations based on observed customer behavior and are not generated from actual campaign performance data.
The current model is based on the available dataset and may require retraining when substantially different customer data is introduced.
Future Enhancements

Potential future improvements include:

Incorporating additional customer behavioral features.
Testing additional clustering algorithms such as DBSCAN or hierarchical clustering.
Adding automated model retraining.
Integrating real-time customer data.
Adding customer lifetime value analysis.
Adding campaign response tracking.
Connecting the dashboard to a live database.
Developing automated personalized campaign recommendations.
Conclusion

This project demonstrates the application of unsupervised machine learning to e-commerce customer behavior analysis.

The final K-Means clustering model uses numerical and categorical customer characteristics to identify seven distinct customer segments.

After preprocessing the selected features using MinMaxScaler and OneHotEncoder, the final model achieved a Silhouette Score of 0.948270.

The resulting segments were analyzed to understand customer behavior and develop targeted marketing strategies.

The Streamlit dashboard provides an interactive interface for exploring customer segments, behavioral characteristics, model performance, and customer-level data.

Overall, the project demonstrates how data analysis and unsupervised machine learning can transform customer behavior data into actionable business insights.