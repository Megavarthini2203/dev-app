import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.plotting import plot_pca_correlation_graph, plot_decision_regions

# Page configuration
st.set_page_config(page_title="Hotel Booking Data Analysis", page_icon="🏨", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\megav\Downloads\hotel_bookings.csv", encoding='latin1')
    return df

df = load_data()

# Data Cleaning and Feature Extraction
data = df.copy()
data = data.fillna({'children': 0, 'agent': 0, 'country': 'Unknown'})
zero_guests = list(data.loc[data["adults"] + data["children"] + data["babies"] == 0].index)
data.drop(data.index[zero_guests], inplace=True)

# Convert object type columns to string type for label encoding
data.loc[:, "hotel"] = data["hotel"].astype(str)
data.loc[:, "arrival_date_month"] = data["arrival_date_month"].astype(str)
data.loc[:, "meal"] = data["meal"].astype(str)
data.loc[:, "country"] = data["country"].astype(str)
data.loc[:, "market_segment"] = data["market_segment"].astype(str)
data.loc[:, "distribution_channel"] = data["distribution_channel"].astype(str)
data.loc[:, "reserved_room_type"] = data["reserved_room_type"].astype(str)
data.loc[:, "assigned_room_type"] = data["assigned_room_type"].astype(str)
data.loc[:, "deposit_type"] = data["deposit_type"].astype(str)
data.loc[:, "customer_type"] = data["customer_type"].astype(str)

# Handle date parsing with specific format
data["reservation_status_date"] = pd.to_datetime(data["reservation_status_date"], format='%m/%d/%Y', errors='coerce')

# Label encoding
label_encoder = LabelEncoder()
columns_to_encode = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                     'deposit_type', 'reservation_status', 'distribution_channel', 'reserved_room_type',
                     'assigned_room_type', 'customer_type']
for col in columns_to_encode:
    data[col] = label_encoder.fit_transform(data[col])

# Remove outliers function
def remove_outliers_binning(data, num_bins=10):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_cleaned = data.copy()
    data_cleaned[(data_cleaned < lower_bound) | (data_cleaned > upper_bound)] = np.nan
    data_cleaned.fillna(data_cleaned.mean(), inplace=True)
    return data_cleaned

# Remove outliers
data_cleaned = data.apply(remove_outliers_binning, axis=0)

# Handle any remaining NaN values
data_cleaned = data_cleaned.fillna(data_cleaned.mean())

# Add background
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://www.example.com/background.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("Hotel Booking Data Analysis")
st.write("Explore and analyze hotel booking data through various visualizations and statistics.")

# Sidebar
st.sidebar.title('Exercises')
selected_exercise = st.sidebar.selectbox('Select an exercise:', 
                                         ['Exercise 1: Choosing the right chart types',
                                          'Exercise 2: Arranging elements for clarity',
                                          'Exercise 3: Employing color schemes strategically',
                                          'Exercise 4: Incorporating interactivity',
                                          'Exercise 5: Conduct exploratory data analysis using visualization',
                                          'Exercise 6: Craft visual presentations of data for effective communication',
                                          'Exercise 7: Use knowledge of perception and cognition to evaluate visualization design alternatives',
                                          'Exercise 8: Design and evaluate color palettes for visualization based on principles of perception',
                                          'Exercise 9: Apply data transformations such as aggregation and filtering for visualization'])

# Based on the selected exercise, display relevant plots or information
if selected_exercise == 'Exercise 1: Choosing the right chart types':
    st.write("Plots related to Exercise 1")
    # Add code to display plots related to Exercise 1
    # Display dataset
    st.subheader("Dataset")
    st.write(data_cleaned.head())

    # Data Overview
    st.subheader("Data Overview")
    st.write(data_cleaned.info())
    st.write(data_cleaned.describe())

    # Checking For Missing Values
    st.subheader("Missing Values")
    missing_values = data_cleaned.isna().sum()
    missing_percentage = (data_cleaned.isna().sum() / data_cleaned.shape[0]) * 100
    missing_percentage_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage.round(2)})
    missing_percentage_df = missing_percentage_df[missing_percentage_df['Percentage'] > 0].sort_values('Percentage', ascending=False)
    st.write(f'Columns with missing values: \n{missing_percentage_df}')
    st.write(f'Total missing values: {data_cleaned.isnull().values.sum()}')

    # Irregular Cardinalities
    st.subheader("Irregular Cardinalities")
    st.write(data_cleaned.nunique())
    st.write("Value counts for 'arrival_date_month':")
    st.write(data_cleaned["arrival_date_month"].value_counts())
    st.write("Value counts for 'arrival_date_year':")
    st.write(data_cleaned["arrival_date_year"].value_counts())

    # Outliers
    st.subheader("Outliers")
    numeric_columns = data_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x=data_cleaned[column], ax=ax)
        ax.set_title(f'Boxplot of {column}')
        st.pyplot(fig)

elif selected_exercise == 'Exercise 2: Arranging elements for clarity':
    st.write("Plots related to Exercise 2")
    # Add code to display plots related to Exercise 2
    # Feature Extraction and PCA
    st.subheader("Feature Extraction and PCA")
    float_columns = data_cleaned.select_dtypes(include=['float']).columns
    X = data_cleaned[float_columns]
    y = data_cleaned['is_canceled']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure there are no NaN or infinite values after scaling
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=None, neginf=None)

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance ratio
    st.write("Explained variance ratio:", pca.explained_variance_ratio_)
    fig, ax = plt.subplots()
    ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='skyblue')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('Explained Variance Ratio by Principal Component')
    st.pyplot(fig)

    # Top features for each principal component
    abs_components = np.abs(pca.components_)
    top_feature_indices = np.argsort(abs_components, axis=1)[:, ::-1]
    top_feature_names = np.array(X.columns)[top_feature_indices]
    for i, component in enumerate(top_feature_names):
        st.write(f"Top features for Principal Component {i}: {component}")

    # Visualizations
    st.subheader("Visualizations")
    st.write("Booking Cancellations by Hotel Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=data_cleaned, x='is_canceled')
    ax.set_title('Cancellation Distribution')
    st.pyplot(fig)

    st.write("Scatter Plot: Previous Cancellations vs Lead Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data_cleaned, x='previous_cancellations', y='lead_time', hue='is_canceled', palette='viridis')
    ax.set_title('Previous Cancellations vs Lead Time with Canceled')
    ax.set_xlabel('Previous Cancellations')
    ax.set_ylabel('Lead Time')
    st.pyplot(fig)

    st.write("Bar Plot: Average Cancellation Due to Children")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data_cleaned, x='children', y='is_canceled', estimator=np.mean, palette='rocket')
    ax.set_title('Average Cancellation Due to Children')
    ax.set_xlabel('Children')
    ax.set_ylabel('Average Cancellation')
    st.pyplot(fig)

    st.write("Box Plot: Cancellation by Previous Cancellations")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=data_cleaned, x='previous_cancellations', y='is_canceled', palette='husl')
    ax.set_title('Cancellation by Previous Cancellations')
    ax.set_xlabel('Previous Cancellations')
    ax.set_ylabel('Is Canceled')
    st.pyplot(fig)

    # 3D Scatter Plot
    st.write("3D Scatter Plot: ADR, Country, Lead Time")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_cleaned['adr'], data_cleaned['country'], data_cleaned['lead_time'], color='red')
    ax.set_xlabel('ADR')
    ax.set_ylabel('Country')
    ax.set_zlabel('Lead Time')
    ax.set_title('ADR, Country, Lead Time')
    st.pyplot(fig)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    correlation_matrix = data_cleaned.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

    # Cancellation Statistics
    st.subheader("Cancellation Statistics")
    total_cancelations = data['is_canceled'].sum()
    rh_cancelations = data.loc[data["hotel"] == "Resort Hotel"]["is_canceled"].sum()
    ch_cancelations = data.loc[data["hotel"] == "City Hotel"]["is_canceled"].sum()
    rel_cancel = (total_cancelations / data.shape[0]) * 100
    rh_rel_cancel = (rh_cancelations / data.loc[data["hotel"] == "Resort Hotel"].shape[0]) * 100
    ch_rel_cancel = (ch_cancelations / data.loc[data["hotel"] == "City Hotel"].shape[0]) * 100
    st.write(f'Total number of cancellations: {total_cancelations}')
    st.write(f'Total number of resort hotel cancellations: {rh_cancelations}')
    st.write(f'Total number of city hotel cancellations: {ch_cancelations}')
    st.write(f'Relative number of cancellations: {rel_cancel:.2f}%')
    st.write(f'Relative number of resort hotel cancellations: {rh_rel_cancel:.2f}%')
    st.write(f'Relative number of city hotel cancellations: {ch_rel_cancel:.2f}%')

    # Model: K-Nearest Neighbors
    st.subheader("K-Nearest Neighbors Model")
    st.write("Training a KNN model to predict booking cancellations based on the features.")
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    st.write(f'KNN Model Accuracy: {accuracy:.2f}')

    # Visualization of PCA Components
    st.write("Visualization of PCA Components")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_pca_correlation_graph(X_pca, y, ax=ax)
    st.pyplot(fig)

    st.write("Decision Regions with KNN")
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_decision_regions(X_pca, y.values, clf=knn, legend=2, ax=ax)
    st.pyplot(fig)

elif selected_exercise == 'Exercise 4: Incorporating interactivity':
    st.write("Plots related to Exercise 4")
    # Exercise 4: New Data Visualizations
    # Pie chart: Distribution of market segments
    st.write("### Market Segment Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    market_segment_counts = data_cleaned['market_segment'].value_counts()
    ax.pie(market_segment_counts, labels=market_segment_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    ax.set_title('Market Segment Distribution')
    st.pyplot(fig)

elif selected_exercise == 'Exercise 5: Conduct exploratory data analysis using visualization':
    st.write("Plots related to Exercise 5")
    # Exercise 5: Exploratory Data Analysis
    # Univariate Analysis: Distribution of Lead Time
    st.write("### Distribution of Lead Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data_cleaned['lead_time'], bins=30, kde=True, color='blue', ax=ax)
    ax.set_title('Distribution of Lead Time')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Bivariate Analysis: Lead Time vs ADR
    st.write("### Lead Time vs ADR")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data_cleaned, x='lead_time', y='adr', hue='is_canceled', palette='coolwarm', ax=ax)
    ax.set_title('Lead Time vs ADR')
    ax.set_xlabel('Lead Time')
    ax.set_ylabel('ADR')
    st.pyplot(fig)

    # Time-Series Analysis: Bookings Over Time
    st.write("### Bookings Over Time")
    data_cleaned['arrival_date'] = pd.to_datetime(data_cleaned['arrival_date_year'].astype(str) + '-' + data_cleaned['arrival_date_month'].astype(str) + '-01')
    monthly_bookings = data_cleaned.groupby('arrival_date').size()
    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_bookings.plot(ax=ax, color='green', marker='o')
    ax.set_title('Bookings Over Time')
    ax.set_xlabel('Arrival Date')
    ax.set_ylabel('Number of Bookings')
    st.pyplot(fig)

    # Missing Data Analysis: Missing Value Heatmap
    st.write("### Missing Value Heatmap")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data_cleaned.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
    ax.set_title('Missing Value Heatmap')
    st.pyplot(fig)

    # Data Transformation: Log Transformation of ADR
    st.write("### Log Transformation of ADR")
    data_cleaned['log_adr'] = np.log1p(data_cleaned['adr'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data_cleaned['log_adr'], bins=30, kde=True, color='purple', ax=ax)
    ax.set_title('Log Transformation of ADR')
    ax.set_xlabel('Log(ADR)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Data Segmentation: Hotel Type and Cancellations
    st.write("### Hotel Type and Cancellations")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data_cleaned, x='hotel', hue='is_canceled', palette='Set1', ax=ax)
    ax.set_title('Hotel Type and Cancellations')
    ax.set_xlabel('Hotel Type')
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif selected_exercise == 'Exercise 6: Craft visual presentations of data for effective communication':
    st.write("Plots related to Exercise 6")
    # Exercise 6: Craft Visual Presentations
    # Highlighting Key Insights: Cancellation Rate by Customer Type
    st.write("### Cancellation Rate by Customer Type")
    cancellation_rate_by_customer_type = data_cleaned.groupby('customer_type')['is_canceled'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    cancellation_rate_by_customer_type.plot(kind='bar', color='dodgerblue', ax=ax)
    ax.set_title('Cancellation Rate by Customer Type')
    ax.set_xlabel('Customer Type')
    ax.set_ylabel('Cancellation Rate')
    st.pyplot(fig)

    # Providing Context and Interpretation: ADR by Market Segment
    st.write("### ADR by Market Segment")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data_cleaned, x='market_segment', y='adr', palette='coolwarm', ax=ax)
    ax.set_title('ADR by Market Segment')
    ax.set_xlabel('Market Segment')
    ax.set_ylabel('ADR')
    st.pyplot(fig)

elif selected_exercise == 'Exercise 7: Use knowledge of perception and cognition to evaluate visualization design alternatives':
    st.write("Plots related to Exercise 7")
    # Exercise 7: Evaluate Visualization Design Alternatives
    # Color Theory: ADR Distribution by Hotel Type
    st.write("### ADR Distribution by Hotel Type")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data_cleaned, x='adr', hue='hotel', multiple='stack', palette='viridis', ax=ax)
    ax.set_title('ADR Distribution by Hotel Type')
    ax.set_xlabel('ADR')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Pre-attentive Processing: Cancellation Rates Over Time
    #st.write("### Cancellation Rates Over Time")
    '''cancellation_rates_over_time = data_cleaned.groupby('arrival_date')['is_canceled'].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    cancellation_rates_over_time.plot(ax=ax, color='red', marker='o')
    ax.set_title('Cancellation Rates Over Time')
    ax.set_xlabel('Arrival Date')
    ax.set_ylabel('Cancellation Rate')
    st.pyplot(fig)'''
    # Combine year, month, and day columns into a single datetime column
    data_cleaned['arrival_date'] = pd.to_datetime(data_cleaned[['arrival_date_year', 'arrival_date_month', 'arrival_date_day']])
    # Group by arrival date and calculate cancellation rates
    cancellation_rates_over_time = data_cleaned.groupby('arrival_date')['is_canceled'].mean()
    # Plot cancellation rates over time
    st.write("Cancellation Rates Over Time")
    st.line_chart(cancellation_rates_over_time)

elif selected_exercise == 'Exercise 8: Design and evaluate color palettes for visualization based on principles of perception':
    st.write("Plots related to Exercise 8")
    # Categorical Palette Example
    st.write("Categorical Palette Example")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x='is_canceled', palette='Set1', ax=ax)
    ax.set_title('Cancellation Distribution (Set1)')
    st.pyplot(fig)

    # Sequential Palette Example
    st.write("Sequential Palette Example")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='lead_time', kde=True, color='Blues', ax=ax)
    ax.set_title('Lead Time Distribution (Blues)')
    st.pyplot(fig)

    # Diverging Palette Example
    st.write("Diverging Palette Example")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x='lead_time', y='adr', hue='is_canceled', palette='RdBu', ax=ax)
    ax.set_title('Lead Time vs ADR (RdBu)')
    st.pyplot(fig)

    # High Contrast Example
    st.write("High Contrast Example")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x='market_segment', y='adr', palette='coolwarm', ax=ax)
    ax.set_title('ADR by Market Segment (coolwarm)')
    st.pyplot(fig)

    # Consistent Color Example
    st.write("Consistent Color Example")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=data, x='adr', kde=True, color='green', ax=ax)
    ax.set_title('ADR Distribution (green)')
    st.pyplot(fig)

    # Handling Missing Data
    st.subheader("Handling Missing Data with Colors")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis', yticklabels=False, ax=ax)
    ax.set_title('Missing Value Heatmap')
    st.pyplot(fig)

elif selected_exercise == 'Exercise 9: Apply data transformations such as aggregation and filtering for visualization':
    st.write("Plots related to Exercise 9")
    # Aggregation and Filtering
    st.subheader("Aggregation and Filtering")
    # Aggregation: Summarization
    st.write("### Aggregation: Summarization")
    monthly_bookings = data.groupby(['arrival_date_year', 'arrival_date_month']).size().reset_index(name='bookings')
    monthly_bookings['arrival_date'] = pd.to_datetime(monthly_bookings['arrival_date_year'].astype(str) + '-' + monthly_bookings['arrival_date_month'].astype(str) + '-01')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_bookings, x='arrival_date', y='bookings', marker='o', ax=ax)
    ax.set_title('Monthly Bookings')
    st.pyplot(fig)

    # Filtering: Subset Selection
    st.write("### Filtering: Subset Selection")
    filtered_data = data[(data['lead_time'] <= 100) & (data['adr'] <= 200)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=filtered_data, x='lead_time', y='adr', hue='is_canceled', palette='coolwarm', ax=ax)
    ax.set_title('Filtered Lead Time vs ADR')
    st.pyplot(fig)

    # Temporal Aggregation: Time Interval Aggregation
    st.write("### Temporal Aggregation: Time Interval Aggregation")
    weekly_bookings = data.resample('W', on='reservation_status_date').size()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=weekly_bookings, ax=ax)
    ax.set_title('Weekly Bookings')
    st.pyplot(fig)

    # Data Joining and Merging
    st.write("### Data Joining and Merging")
    additional_data = pd.DataFrame({'customer_id': np.arange(1, 101), 'loyalty_points': np.random.randint(100, 500, size=100)})
    merged_data = pd.merge(data, additional_data, how='left', left_on='customer_type', right_on='customer_id')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=merged_data, x='lead_time', y='loyalty_points', hue='is_canceled', palette='viridis', ax=ax)
    ax.set_title('Lead Time vs Loyalty Points')
    st.pyplot(fig)



