import streamlit as st
import streamlit.components.v1 as components

# Function to inject CSS for aesthetics
def add_custom_css():
    custom_css = """
    <style>
        body {
            background-color: #f0f2f6; /* Light background */
            font-family: 'Arial', sans-serif; /* Set a professional font */
        }
        .stButton button {
            background-color: #007ACC; /* Primary button color */
            color: white;
            border-radius: 5px;
            padding: 10px 20px; /* Increased padding for better appearance */
            font-size: 16px;
            transition: all 0.3s ease;
            border: none; /* Remove border for a cleaner look */
        }
        .stButton button:hover {
            background-color: #005A9E; /* Darker shade on hover */
        }
        .header-section {
            text-align: center;
            margin-bottom: 20px;
        }
        .login-container {
            width: 300px; /* Fixed width for better alignment */
            margin: auto; /* Center the container */
            padding: 20px; /* Add padding for aesthetics */
            border: 1px solid #d1d1d1; /* Subtle border */
            border-radius: 10px; /* Rounded corners */
            background-color: white; /* White background for contrast */
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); /* Subtle shadow */
        }
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #888;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def login_page():
    # Add the application title and description in the login page itself
    st.markdown('<div class="header-section"><h1>Supermarket DSS Application</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="header-section"><h3>Welcome to the Decision Support System for Supermarkets</h3></div>', unsafe_allow_html=True)
    
    st.subheader("Please enter your credentials")

    # Input fields for username and password
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")

    # Check if the login button is pressed
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True  # Set a session state variable
            st.session_state.current_page = "Sales"  # Redirect to default module
            st.success("Login successful! Welcome to the dashboard.")
            dashboard_layout()
        else:
            st.error("Invalid username or password")

# Function for dashboard layout with navigation
def dashboard_layout():
    st.sidebar.title("DSS Navigation")

    # Sidebar module selection with icons
    module = st.sidebar.selectbox(
        "Select Module",
        ["Sales", "Campaign", "Survey", "Inventory", "Customer"],
        format_func=lambda x: {
            "Sales": "📊 Sales",
            "Campaign": "📢 Campaign",
            "Survey": "📝 Survey",
            "Inventory": "📦 Inventory",
            "Customer": "👥 Customer"
        }.get(x, x)
    )

    # Show the respective page based on the sidebar selection
    if module == "Sales":
        sales_module()  # Placeholder for the sales module
    elif module == "Campaign":
        campaign_module()  # Placeholder for the campaign module
    elif module == "Survey":
        survey_module()  # Placeholder for the survey module
    elif module == "Inventory":
        inventory_module()  # Placeholder for the inventory module
    elif module == "Customer":
        customer_module()  # Placeholder for the customer module

    # Add a logout button in the sidebar
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_page = "Login"
        st.info("You have been logged out. Please log in again.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### © 2024 Supermarket DSS Application")
    st.sidebar.markdown("For support, contact: support@example.com")


def run_app():
    add_custom_css()  # Inject custom CSS

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False  # Initialize logged_in state

    # Set the default module to Sales if logged in
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Sales"  # Set default page to Sales

    # Check login state
    if not st.session_state.logged_in:
        login_page()  # Show login page
    else:
        # Render dashboard layout
        dashboard_layout()  # Get the selected module


def campaign_module():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

    st.title("Campaign Analysis Dashboard")

    # Load Data
    @st.cache_data
    def load_data():
        return pd.read_csv('campaign_data.csv')  # Replace with your actual data file path

    campaign_df = load_data()

    # Convert StartDate and EndDate to datetime format
    campaign_df['StartDate'] = pd.to_datetime(campaign_df['StartDate'])
    campaign_df['EndDate'] = pd.to_datetime(campaign_df['EndDate'])

    # Remove StartDate, EndDate, and CampaignID from analysis
    analysis_df = campaign_df.drop(columns=['StartDate', 'EndDate', 'CampaignID'])

    # Initialize scaler for normalization
    scalers = {}
    for col in ['Budget', 'RevenueGenerated']:
        scaler = StandardScaler()
        analysis_df[col] = scaler.fit_transform(analysis_df[[col]])
        scalers[col] = scaler

 
    sections = ["Univariate Analysis", "Bivariate Analysis", "Predictive Modeling", 
                "Trends Over Time", "Channel Performance", "Target Audience Performance", 
                "Budget Efficiency"]
    selected_section = st.sidebar.selectbox("Select a section", sections)

    # Univariate Analysis
    if selected_section == "Univariate Analysis":
        st.subheader("Univariate Analysis")
        columns = st.multiselect("Select columns for Univariate Analysis", list(analysis_df.columns), default=list(analysis_df.columns))

        for column in columns:
            st.write(f"Analysis for column: {column}")
            if analysis_df[column].dtype == 'object':
                st.write(analysis_df[column].value_counts())
                fig = px.histogram(analysis_df, x=column)
                st.plotly_chart(fig)
            else:
                st.write(analysis_df[column].describe())
                fig = px.histogram(analysis_df, x=column, marginal="box", nbins=30)
                st.plotly_chart(fig)

    # Bivariate Analysis
    elif selected_section == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")
        numeric_columns = analysis_df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = analysis_df.select_dtypes(include=['object']).columns.tolist()

        for num_col in numeric_columns:
            for cat_col in categorical_columns:
                fig = px.box(analysis_df, x=cat_col, y=num_col, points="all")
                st.plotly_chart(fig)

    # Predictive Modeling
    elif selected_section == "Predictive Modeling":
        st.subheader("Predictive Modeling")

        target = st.selectbox("Select Target Variable", analysis_df.columns)
        input_features = analysis_df.drop(columns=[target]).columns

        # Encode categorical features
        X = analysis_df.drop(columns=[target])
        y = analysis_df[target]

        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                label_encoders[col] = LabelEncoder()
                X[col] = label_encoders[col].fit_transform(X[col])

        if y.dtype == 'object':
            label_encoders[target] = LabelEncoder()
            y = label_encoders[target].fit_transform(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Model training
        if y.dtype in [np.int64, np.float64]:
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
        else:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")

        # Prediction Section
        st.subheader("Predict Target Variable")
        input_values = {}
        for feature in input_features:
            unique_values = analysis_df[feature].unique()
            if analysis_df[feature].dtype == 'object' and len(unique_values) > 0:
                input_values[feature] = st.selectbox(f"Select {feature}", unique_values)
            else:
                input_values[feature] = st.number_input(f"Input {feature}", value=np.nan)

        for feature in input_features:
            if pd.isna(input_values[feature]) or input_values[feature] == "":
                if analysis_df[feature].dtype == 'object':
                    input_values[feature] = analysis_df[feature].mode()[0]
                else:
                    input_values[feature] = analysis_df[feature].mean()

        input_df = pd.DataFrame([input_values])

        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)

        if target in ['Budget', 'RevenueGenerated']:
            prediction = np.array(prediction).reshape(-1, 1)
            prediction = scalers[target].inverse_transform(prediction)
            prediction = prediction[0][0]

        if target in label_encoders:
            prediction = label_encoders[target].inverse_transform([int(prediction)])[0]

        st.write(f"Predicted {target}: {prediction}")

    # Trends Over Time
    elif selected_section == "Trends Over Time":
        st.subheader("Trends Over Time")
        # Trends for Revenue Generated
        revenue_trends = campaign_df.groupby(campaign_df['StartDate'].dt.to_period('M')).agg({'RevenueGenerated': 'sum'}).reset_index()
        revenue_trends['StartDate'] = revenue_trends['StartDate'].dt.to_timestamp()
        fig_revenue = px.line(revenue_trends, x='StartDate', y='RevenueGenerated', title='Total Revenue Generated Over Time', markers=True)
        st.plotly_chart(fig_revenue)

        # Trends for Budget
        budget_trends = campaign_df.groupby(campaign_df['StartDate'].dt.to_period('M')).agg({'Budget': 'mean'}).reset_index()
        budget_trends['StartDate'] = budget_trends['StartDate'].dt.to_timestamp()
        fig_budget = px.line(budget_trends, x='StartDate', y='Budget', title='Average Budget Spent Over Time', markers=True)
        st.plotly_chart(fig_budget)

            
        # Ensure proper handling of the RevenueGenerated and Budget columns
        revenue_mean = campaign_df['RevenueGenerated'].mean()
        budget_mean = campaign_df['Budget'].mean()

        # Calculate quantiles for better insights
        revenue_quantile_25 = campaign_df['RevenueGenerated'].quantile(0.25)
        revenue_quantile_75 = campaign_df['RevenueGenerated'].quantile(0.75)
        budget_quantile_25 = campaign_df['Budget'].quantile(0.25)

        # Generate insights based on different conditions
        if revenue_mean < revenue_quantile_25:
            st.warning("The mean Revenue Generated is below the 25th percentile. Consider investigating the factors contributing to this, such as targeting, channels, or campaign execution.")

        if budget_mean > revenue_mean:
            st.warning("The average Budget exceeds the average Revenue Generated. Consider optimizing campaign spending and reallocating resources to improve return on investment.")

        # Additional insights based on budget and revenue distribution
        if budget_quantile_25 > revenue_quantile_75:
            st.warning("The 25th percentile of Budget is higher than the 75th percentile of Revenue Generated. This suggests a disparity where lower-performing campaigns may be receiving higher budgets. Review and reallocate budgets accordingly.")

        if revenue_quantile_75 < budget_mean:
            st.warning("The 75th percentile of Revenue Generated is lower than the average Budget. High-budget campaigns are not necessarily generating high revenue. Investigate the effectiveness of high-budget campaigns.")

        # You might also want to add a more general insight if no specific conditions are met
        if revenue_mean >= revenue_quantile_25 and budget_mean <= revenue_mean:
            st.info("The current campaign strategy seems balanced with Revenue Generated generally aligning with Budget. Continue monitoring and optimizing for better performance.")



    # Channel Performance
    elif selected_section == "Channel Performance":
        st.subheader("Channel Performance")
        channel_trends = campaign_df.groupby(['StartDate', 'Channel']).agg({'RevenueGenerated': 'mean'}).reset_index()
        channel_trends['StartDate'] = pd.to_datetime(channel_trends['StartDate'])
        fig_channel = px.line(channel_trends, x='StartDate', y='RevenueGenerated', color='Channel', title='Revenue Generated by Channel Over Time', markers=True)
        st.plotly_chart(fig_channel)

        channel_performance = campaign_df.groupby('Channel').agg({'RevenueGenerated': 'mean'}).sort_values(by='RevenueGenerated', ascending=False)
        st.write(channel_performance)

        top_channel = channel_performance.index[0]
        bottom_channel = channel_performance.index[-1]

        st.write(f"The top-performing channel is: {top_channel} with an average revenue of {channel_performance.loc[top_channel, 'RevenueGenerated']:.2f}.")
        st.write(f"The lowest-performing channel is: {bottom_channel} with an average revenue of {channel_performance.loc[bottom_channel, 'RevenueGenerated']:.2f}.")

        if channel_performance.loc[bottom_channel, 'RevenueGenerated'] < channel_performance['RevenueGenerated'].mean():
            st.warning(f"The {bottom_channel} channel is underperforming compared to the average. Consider revising strategies or reallocating budget to more effective channels.")

    # Target Audience Performance
    elif selected_section == "Target Audience Performance":
        st.subheader("Target Audience Performance")
        audience_trends = campaign_df.groupby(['StartDate', 'TargetAudience']).agg({'RevenueGenerated': 'mean'}).reset_index()
        audience_trends['StartDate'] = pd.to_datetime(audience_trends['StartDate'])
        fig_audience = px.line(audience_trends, x='StartDate', y='RevenueGenerated', color='TargetAudience', title='Revenue Generated by Target Audience Over Time', markers=True)
        st.plotly_chart(fig_audience)

        audience_performance = campaign_df.groupby('TargetAudience').agg({'RevenueGenerated': 'mean'}).sort_values(by='RevenueGenerated', ascending=False)
        st.write(audience_performance)

        top_audience = audience_performance.index[0]
        bottom_audience = audience_performance.index[-1]

        st.write(f"The most lucrative target audience is: {top_audience} with an average revenue of {audience_performance.loc[top_audience, 'RevenueGenerated']:.2f}.")
        st.write(f"The least lucrative target audience is: {bottom_audience} with an average revenue of {audience_performance.loc[bottom_audience, 'RevenueGenerated']:.2f}.")

        if audience_performance.loc[bottom_audience, 'RevenueGenerated'] < audience_performance['RevenueGenerated'].mean():
            st.warning(f"The {bottom_audience} audience is underperforming compared to the average. Consider adjusting the campaign strategy for this group.")

    # Budget Efficiency
    elif selected_section == "Budget Efficiency":
        st.subheader("Budget Efficiency")
        budget_efficiency = campaign_df.groupby('Channel').agg({'Budget': 'sum', 'RevenueGenerated': 'sum'}).reset_index()
        budget_efficiency['Efficiency'] = budget_efficiency['RevenueGenerated'] / budget_efficiency['Budget']
        fig_budget_efficiency = px.bar(budget_efficiency, x='Channel', y='Efficiency', title='Budget Efficiency by Channel', color='Efficiency', labels={'Efficiency': 'Revenue per Unit Budget'})
        st.plotly_chart(fig_budget_efficiency)

        st.write(budget_efficiency)


        for index, row in budget_efficiency.iterrows():
            if row['Efficiency'] < 1:
                st.warning(f"The {row['Channel']} channel has an efficiency below 1, indicating it is not generating enough revenue to justify the budget spent.")
        campaign_df['BudgetEfficiency'] = campaign_df['RevenueGenerated'] / campaign_df['Budget']
        avg_efficiency = campaign_df['BudgetEfficiency'].mean()

        st.subheader("Budget Allocation Efficiency")
        st.write(campaign_df[['CampaignID', 'Budget', 'RevenueGenerated', 'BudgetEfficiency']])

        if avg_efficiency < 1:
            st.warning("The average budget efficiency is less than 1, indicating that on average, the campaigns are not generating revenue equal to or greater than the budget spent. Consider optimizing the campaigns to improve efficiency.")

        audience_performance = campaign_df.groupby('TargetAudience').agg({'RevenueGenerated': 'mean'}).sort_values(by='RevenueGenerated', ascending=False)



def survey_module():
    import pandas as pd
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    import plotly.express as px

    nltk.download('punkt')
    nltk.download('stopwords')

    # Load Data
    @st.cache_data
    def load_data():
        return pd.read_csv('survey_data.csv')  # Replace with your file path

    survey_df = load_data()

    # Convert Date to datetime format for time-based analysis
    survey_df['Date'] = pd.to_datetime(survey_df['Date'])

    # Title
    st.title("Survey Analysis Dashboard")

    # Rating Distribution
    st.subheader("Rating Distribution")
    rating_counts = survey_df['Rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)

    # Filter by Rating
    st.subheader("Filter Reviews by Rating")
    selected_rating = st.selectbox("Select Rating", survey_df['Rating'].unique())
    filtered_data = survey_df[survey_df['Rating'] == selected_rating]
    st.write(filtered_data)

    # Positive and Negative Reviews Percentage
    st.subheader("Positive and Negative Reviews Percentage")
    # Calculate positive and negative reviews
    positive_reviews = survey_df[survey_df['Rating'] >= 4]
    negative_reviews = survey_df[survey_df['Rating'] <= 2]

    # Create a DataFrame for pie chart data
    pie_data = pd.DataFrame({'Review Type': ['Positive', 'Negative'],
                             'Count': [len(positive_reviews), len(negative_reviews)]})

    # Create the pie chart
    fig = px.pie(pie_data, values='Count', names='Review Type', title='Review Sentiment')

    # Make the chart interactive
    fig.update_traces(hoverinfo='label+value', textinfo='percent+value', textfont_size=15,
                      marker=dict(line=dict(color='#000000', width=2)))

    st.plotly_chart(fig)

    # Add a dropdown for selecting review type
    st.subheader("Select Review Type to View Comments")
    review_type = st.selectbox("Select Review Type", ["Positive", "Negative"])

    # Show reviews based on selected review type
    if review_type == 'Positive':
        st.write(positive_reviews)
    elif review_type == 'Negative':
        st.write(negative_reviews)

    # Trends Over Time
    st.subheader("Trends Over Time")
    trends_df = survey_df.groupby(survey_df['Date'].dt.to_period('M')).agg({'Rating': 'mean'}).reset_index()
    trends_df['Date'] = trends_df['Date'].dt.to_timestamp()

    fig = px.line(trends_df, x='Date', y='Rating', title='Average Rating Over Time', markers=True)
    fig.update_layout(yaxis_title='Average Rating', xaxis_title='Date')

    st.plotly_chart(fig)

    # Insights
    st.subheader("Actionable Insights")

    if trends_df['Rating'].iloc[-1] < trends_df['Rating'].mean():
        st.warning("Recent ratings are below the average. Investigate potential issues in customer satisfaction.")
    else:
        st.success("Recent ratings are above the average. Customer satisfaction seems to be improving.")

    if len(negative_reviews) / len(survey_df) > 0.3:
        st.warning("More than 30% of the reviews are negative. Consider reviewing the common complaints.")

    common_negative_issues = ['price', 'wait', 'service']
    for issue in common_negative_issues:
        if len(negative_reviews[negative_reviews['Comments'].str.contains(issue, case=False, na=False)]) > 10:
            st.info(f"Common issue detected: Customers frequently mention '{issue}'. Consider addressing this.")

    # Most Common Words Filter
    st.subheader("Filter Comments by Common Words")

    # Tokenize and filter words
    all_comments = ' '.join(filtered_data['Comments'].dropna())
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(all_comments.lower())
    filtered_words = [w for w in word_tokens if w.isalpha() and w not in stop_words]

    # Calculate word frequencies
    word_freq = pd.Series(filtered_words).value_counts().head(20)
    selected_word = st.selectbox("Select a Word", word_freq.index)

    # Show Comments Containing Selected Word
    st.subheader(f"Comments Containing '{selected_word}'")
    matching_comments = filtered_data[filtered_data['Comments'].str.contains(selected_word, case=False, na=False)]
    st.write(matching_comments)



def inventory_module():
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import plotly.graph_objects as go
    from statsmodels.tsa.api import ExponentialSmoothing
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from babel.numbers import format_currency

    today = pd.Timestamp.today()

    # Load data
    @st.cache_data
    def load_data():
        try:
            products_df = pd.read_csv("products.csv")  # Replace with your actual data path
            return products_df
        except FileNotFoundError as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()

    products_df = load_data()

    def load_sales_data():
        try:
            sales_df = pd.read_csv("sales.csv")  # Replace with your actual data path
            sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%Y-%m-%d')
            return sales_df
        except FileNotFoundError as e:
            st.error(f"Error loading sales data: {e}")
            return pd.DataFrame()

    sales_df = load_sales_data()

    def forecast_sales(product_id, history, forecast_days=30):
        model = ExponentialSmoothing(history, seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(forecast_days)
        return forecast  # Ensure this is a Series or 1D array

    # Function to calculate overall model accuracy
    def calculate_accuracy(product_id, history, forecast_days=30):
        forecast = forecast_sales(product_id, history, forecast_days)
        actual = history[-forecast_days:]  # Compare with the last part of the history
        if len(actual) == len(forecast):
            mae = mean_absolute_error(actual, forecast)
            mse = mean_squared_error(actual, forecast)
            return mae, mse
        return None, None


    selected_category = st.sidebar.multiselect(
        "Select Product Category", options=products_df['Category'].unique(), default=products_df['Category'].unique()
    )
    selected_brand = st.sidebar.multiselect(
        "Select Brand", options=products_df['Brand'].unique(), default=products_df['Brand'].unique()
    )
    selected_supplier = st.sidebar.multiselect(
        "Select Supplier", options=products_df['SupplierName'].unique(), default=products_df['SupplierName'].unique()
    )

    filtered_data = products_df[
        (products_df['Category'].isin(selected_category)) &
        (products_df['Brand'].isin(selected_brand)) &
        (products_df['SupplierName'].isin(selected_supplier))
    ]

    st.title("Inventory Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    total_products = filtered_data['ProductID'].nunique()
    total_reorder_value = filtered_data['ReorderLevel'].sum()
    filtered_data['StockValue'] = filtered_data['StockLevel'] * filtered_data['Price']
    total_stock_value = filtered_data['StockValue'].sum()
    total_stock = filtered_data['StockLevel'].sum()
    formatted_stock_value = format_currency(total_stock_value, 'INR', locale='en_IN').split('.')[0]
    formatted_reorder_value = f"₹{total_reorder_value:,.0f}"

    with col1:
        st.image("stock_icon1.png", width=80)
        st.metric(label="Total Stock", value=f"{total_stock:,} units")

    with col2:
        st.image("stock_icon.png", width=80)
        st.metric(label="Total Stock Value", value=formatted_stock_value)

    with col3:
        st.image("products_icon.png", width=80)
        st.metric(label="Total Products", value=total_products)

    # Function to forecast sales and check stock arrival
    def forecast_and_check_stock(product_id, history, forecast_days=30, days_for_stock_arrival=0):
        forecast = forecast_sales(product_id, history, forecast_days)
        total_forecasted_sales = forecast.sum()
        return total_forecasted_sales, total_forecasted_sales + history.sum() - forecast_days

    # Analyzing Products Below Reorder Level
    st.header("Products Below Reorder Level")

    below_reorder_data = filtered_data[filtered_data['StockLevel'] < filtered_data['ReorderLevel']]

    if not below_reorder_data.empty:
        def forecast_for_product(row):
            product_id = row['ProductID']
            history = sales_df[sales_df['ProductID'] == product_id].set_index('Date').resample('D').sum()['Quantity'].fillna(0)
            forecast = forecast_sales(product_id, history, forecast_days=row['Days_for_stock_arrival'])
            return forecast.sum()  # Sum the forecasted sales to get the total forecasted sales

        below_reorder_data['ForecastedSales'] = below_reorder_data.apply(forecast_for_product, axis=1)
        below_reorder_data['ExpectedRemainingStock'] = below_reorder_data['StockLevel'] - below_reorder_data['ForecastedSales']
        
        # Filter products that may not sell out before stock arrival
        unsold_products = below_reorder_data[below_reorder_data['ExpectedRemainingStock'] > 0]
        # Filter products that may sell out before stock arrival
        sold_products = below_reorder_data[below_reorder_data['ExpectedRemainingStock'] <= 0]

        if not sold_products.empty:
            st.warning(f"{len(sold_products)} products are expected to sell out before stock arrival.")
            st.dataframe(sold_products[['ProductID', 'ProductName', 'StockLevel', 'ReorderLevel', 'ForecastedSales', 'Days_for_stock_arrival']])

        if not unsold_products.empty:
            st.success(f"{len(unsold_products)} products may not sell out before stock arrival.")
            st.dataframe(unsold_products[['ProductID', 'ProductName', 'StockLevel', 'ReorderLevel', 'ForecastedSales', 'Days_for_stock_arrival']])
        
    else:
        st.success("No products are below the reorder level.")

    # Products Needing Attention Due to Shelf Life
    st.header("Products Needing Attention Due to Shelf Life")

    products_df['ExpiryDate'] = today + pd.to_timedelta(products_df['ShelfLife'], unit='d')
    expiring_products = products_df[products_df['ExpiryDate'] < (today + pd.Timedelta(days=30))]

    if not expiring_products.empty:
        expiring_products['ForecastedSales'] = expiring_products.apply(
            lambda row: forecast_sales(row['ProductID'], 
                                       sales_df[sales_df['ProductID'] == row['ProductID']].set_index('Date').resample('D').sum()['Quantity'].fillna(0), 
                                       forecast_days=(pd.to_datetime(row['ExpiryDate'], format='%d-%m-%Y') - today).days).sum(),
            axis=1
        )
        expiring_products['ExpectedRemainingStock'] = expiring_products['StockLevel'] - expiring_products['ForecastedSales']
        expiring_products['ExpiryDate'] = expiring_products['ExpiryDate'].dt.strftime('%d-%m-%Y')
        
        # Filter products that may not sell out
        unsold_products = expiring_products[expiring_products['ExpectedRemainingStock'] > 0]
        # Filter products that may sell out
        sold_products = expiring_products[expiring_products['ExpectedRemainingStock'] <= 0]
        
        if not unsold_products.empty:
            st.success(f"{len(unsold_products)} products may not sell out before expiry. Consider promotional actions.")
            st.dataframe(unsold_products[['ProductID', 'ProductName', 'ShelfLife', 'ExpiryDate', 'StockLevel', 'ForecastedSales', 'ExpectedRemainingStock']])
        
        if not sold_products.empty:
            st.warning(f"{len(sold_products)} products are expected to sell out before expiry.")
            st.dataframe(sold_products[['ProductID', 'ProductName', 'ShelfLife', 'ExpiryDate', 'StockLevel', 'ForecastedSales']])
    else:
        st.success("No products are nearing their expiry within the next 30 days.")

    # Calculate overall model accuracy
    all_products = products_df['ProductID'].unique()
    all_mae, all_mse = [], []

    for product_id in all_products:
        history = sales_df[sales_df['ProductID'] == product_id].set_index('Date').resample('D').sum()['Quantity'].fillna(0)
        mae, mse = calculate_accuracy(product_id, history, forecast_days=30)
        if mae is not None:
            all_mae.append(mae)
            all_mse.append(mse)

    if all_mae and all_mse:
        avg_mae = np.mean(all_mae)
        avg_mse = np.mean(all_mse)
        st.header("Overall Model Accuracy")
        st.metric("Mean Absolute Error (MAE)", f"{avg_mae:.2f}")
        
    else:
        st.warning("No model accuracy metrics available due to insufficient data.")

    st.header("Inventory per Product Category")

    # Function to plot bar charts
    def plot_bar_chart(data, x, y, title, xlabel, ylabel, rotation=45, currency=False):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=data, x=x, y=y, ci=None, ax=ax)
        for p in ax.patches:
            if p.get_height() > 0:
                formatted_value = f"{p.get_height():,.0f}" if not currency else format_currency(p.get_height(), 'INR', locale='en_IN').split('.')[0]
                ax.annotate(formatted_value, (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right')
        st.pyplot(fig)



    # Grouping and summarizing the data for Category
    category_data = filtered_data.groupby('Category').agg({'StockLevel': 'sum'}).reset_index()
    plot_bar_chart(category_data, 'Category', 'StockLevel', 
                   'Inventory per Product Category', 'Product Category', 'Stock Level')

    st.header("Inventory per Brand")

    # Grouping and summarizing the data for Brand
    brand_data = filtered_data.groupby('Brand').agg({'StockLevel': 'sum'}).reset_index()
    plot_bar_chart(brand_data, 'Brand', 'StockLevel', 
                   'Inventory per Brand', 'Brand', 'Stock Level')

    def plot_pie_chart(data, values_col, labels_col, title):
        fig = go.Figure(data=[go.Pie(labels=data[labels_col], values=data[values_col], hole=0.3)])
        fig.update_layout(title=title)
        st.plotly_chart(fig)
    stock_levels_by_supplier = filtered_data.groupby('SupplierName')['StockLevel'].sum().reset_index()
    stock_value_by_supplier = filtered_data.groupby('SupplierName')['StockValue'].sum().reset_index()
    # Plot Pie Charts
    st.header("Stock Level Distribution by Supplier")
    plot_pie_chart(stock_levels_by_supplier, 'StockLevel', 'SupplierName', 'Stock Level Distribution by Supplier')

    st.header("Stock Value Distribution by Supplier")
    plot_pie_chart(stock_value_by_supplier, 'StockValue', 'SupplierName', 'Stock Value Distribution by Supplier')


def sales_module():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from mlxtend.frequent_patterns import apriori, association_rules
    from babel.numbers import format_currency
    import numpy as np
    import plotly.graph_objects as go

    from statsmodels.tsa.api import ExponentialSmoothing
    from sklearn.model_selection import train_test_split
    # Load data
    @st.cache_data
    def load_data():
        try:
            products_df = pd.read_csv("products.csv")  # Replace with your actual data path
            sales_df = pd.read_csv("sales.csv")        # Replace with your actual data path
            return products_df, sales_df
        except FileNotFoundError as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    products_df, sales_df = load_data()

    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_error
    import numpy as np

    def calculate_accuracy(actual, forecasted):
        mae = mean_absolute_error(actual, forecasted)
        mse = mean_squared_error(actual, forecasted)
        rmse = np.sqrt(mse)
        return mae, mse, rmse


    def forecast_total_revenue(data, days):
        
        daily_revenue = data.groupby('Date').agg({'Total': 'sum'}).reset_index()
        daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
        daily_revenue.set_index('Date', inplace=True)

        # Fit SARIMA Model
        model_revenue = SARIMAX(daily_revenue['Total'], 
                                order=(5, 1, 0),
                                seasonal_order=(1, 1, 1, 7))
        model_revenue_fit = model_revenue.fit()
        
        # Forecast
        forecast_revenue = model_revenue_fit.get_forecast(steps=days)
        forecast_conf_int = forecast_revenue.conf_int()
        forecast_values = forecast_revenue.predicted_mean
        
        # Forecast Dates
        forecast_dates = pd.date_range(start=daily_revenue.index[-1] + pd.Timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedRevenue': forecast_values})
        forecast_df.set_index('Date', inplace=True)

        # Calculate accuracy if historical data for the forecasted period is available
        if len(daily_revenue) > days:
            actual_values = daily_revenue['Total'][-days:]
            forecasted_values = forecast_df['ForecastedRevenue'][:len(actual_values)]
            mae, mse, rmse = calculate_accuracy(actual_values, forecasted_values)
            
        return forecast_df



    sales_df['Date'] = pd.to_datetime(sales_df['Date'], format='%Y-%m-%d')
    sales_products_merged = sales_df.merge(products_df, how="left", on="ProductID")

    

    min_date = sales_products_merged['Date'].min().date()
    max_date = sales_products_merged['Date'].max().date()
    start_date, end_date = st.sidebar.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

    selected_category = st.sidebar.multiselect("Select Product Category", options=sales_products_merged['Category'].unique(), default=sales_products_merged['Category'].unique())
    selected_brand = st.sidebar.multiselect("Select Brand", options=sales_products_merged['Brand'].unique(), default=sales_products_merged['Brand'].unique())

    filtered_data = sales_products_merged[
        (sales_products_merged['Category'].isin(selected_category)) &
        (sales_products_merged['Brand'].isin(selected_brand)) &
        (sales_products_merged['Date'] >= pd.to_datetime(start_date)) &
        (sales_products_merged['Date'] <= pd.to_datetime(end_date))
    ]

    st.title("Sales Analysis Dashboard")

    col1, col2, col3 = st.columns(3)

    total_revenue = filtered_data['Total'].sum()
    total_profit = filtered_data['Profit'].sum()
    total_items_sold = filtered_data['Quantity'].sum()
    formatted_revenue = format_currency(total_revenue, 'INR', locale='en_IN').split('.')[0]
    formatted_profit = format_currency(total_profit, 'INR', locale='en_IN').split('.')[0]


    with col1:
        st.image("revenue_icon.png", width=80)
        # Replace with path to your revenue image
        st.metric(label="Total Revenue ", value=formatted_revenue)

    with col2:
        st.image("profit_icon.png", width=80)  # Replace with path to your profit image
        st.metric(label="Total Profit ", value=formatted_profit)

    with col3:
        st.image("items_sold_icon.png", width=80)  # Replace with path to your items sold image
        st.metric(label="Total Items Sold", value=f"{total_items_sold:,} units")
        
    # Sales per Product Category with annotations
    import matplotlib.pyplot as plt
    import seaborn as sns
    from babel.numbers import format_currency

    st.header("Sales per Product Category")

    # Grouping and summarizing the data
    product_data = filtered_data.groupby('Category')['Total'].sum().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=product_data, x='Category', y='Total', ci=None, ax=ax)

    # Annotating the bars with Indian number formatting
    for p in ax.patches:
        if p.get_height() > 0:
            formatted_value = format_currency(p.get_height(), 'INR', locale='en_IN').split('.')[0]
            ax.annotate(formatted_value, (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Setting titles and labels
    ax.set_title('Sales per Product Category')
    ax.set_xlabel('Product Category')
    ax.set_ylabel('Sales')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Displaying the plot in Streamlit
    st.pyplot(fig)




    # Sales per Brand with annotations
    import matplotlib.pyplot as plt
    import seaborn as sns
    from babel.numbers import format_currency

    st.header("Sales per Brand")

    # Grouping and summarizing the data
    brand_data = filtered_data.groupby('Brand')['Total'].sum().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=brand_data, x='Brand', y='Total', ci=None, ax=ax)

    # Annotating the bars with Indian number formatting
    for p in ax.patches:
        if p.get_height() > 0:
            formatted_value = format_currency(p.get_height(), 'INR', locale='en_IN').split('.')[0]
            ax.annotate(formatted_value, (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Setting titles and labels
    ax.set_title('Sales per Brand')
    ax.set_xlabel('Brand')
    ax.set_ylabel('Sales')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    # Displaying the plot in Streamlit
    st.pyplot(fig)



    # Sales Trend Over Time with annotations
    import matplotlib.pyplot as plt
    import seaborn as sns
    from babel.numbers import format_currency

    st.header("Sales Trend Over Time")
    # Grouping sales by date
    sales_trend = filtered_data.groupby('Date')['Total'].sum().reset_index()

    # Create a Plotly figure
    fig = go.Figure()

    # Add line trace for sales trend
    fig.add_trace(go.Scatter(
        x=sales_trend['Date'],
        y=sales_trend['Total'],
        mode='lines+markers',
        name='Sales Trend',
        text=sales_trend.apply(lambda row: f"Date: {row['Date'].strftime('%d-%m-%Y')}<br>Total Sales: ₹{format_currency(row['Total'], 'INR', locale='en_IN').split('.')[0]}", axis=1),
        hoverinfo='text'
    ))

    # Highlight the peak sales day
    max_sales_date = sales_trend[sales_trend['Total'] == sales_trend['Total'].max()]['Date'].iloc[0]
    max_sales_value = sales_trend['Total'].max()
    formatted_max_sales_value = format_currency(max_sales_value, 'INR', locale='en_IN').split('.')[0]

    fig.add_trace(go.Scatter(
        x=[max_sales_date],
        y=[max_sales_value],
        mode='markers',
        marker=dict(color='red', size=10),
        name='Peak Sales Day',
        text=[f"Peak: {formatted_max_sales_value}"],
        hoverinfo='text'
    ))

    # Update layout
    fig.update_layout(
        title='Sales Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Total Sales',
        xaxis_tickformat='%B',  # Format to display month names
        hovermode='closest',
        width=800,   # Set the width of the plot
        height=600    # Set the height of the plot
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


    import matplotlib.pyplot as plt
    import seaborn as sns
    from babel.numbers import format_currency

    st.header("Sales Trends by Weekday")

    # Adding the 'Weekday' column to the data
    filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()

    # Grouping sales by weekday and reindexing to ensure correct order
    weekday_trend = filtered_data.groupby('Weekday')['Total'].sum().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ).reset_index()

    # Plotting the sales trend by weekday
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=weekday_trend, x='Weekday', y='Total', ci=None, ax=ax)

    # Setting titles and labels
    ax.set_title('Total Sales by Weekday')
    ax.set_xlabel('Weekday')
    ax.set_ylabel('Total Sales')

    # Annotating the bars with Indian number formatting
    for p in ax.patches:
        if p.get_height() > 0:
            formatted_value = format_currency(p.get_height(), 'INR', locale='en_IN').split('.')[0]
            ax.annotate(formatted_value, 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Displaying the plot in Streamlit
    st.pyplot(fig)


    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from babel.numbers import format_currency

    st.header("Overall Sales Trend Insights")

    # Adding relevant columns for analysis
    filtered_data['Day'] = filtered_data['Date'].dt.day
    filtered_data['Weekday'] = filtered_data['Date'].dt.day_name()

    # General Sales Patterns
    overall_trend = filtered_data.groupby('Day')['Total'].sum()
    weekend_sales = filtered_data[filtered_data['Weekday'].isin(['Saturday', 'Sunday'])]['Total'].sum()
    weekday_sales = filtered_data[filtered_data['Weekday'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])]['Total'].sum()

    st.subheader("General Patterns")

    # Calculate average sales per day for weekends and weekdays
    avg_weekend_sales_per_day = weekend_sales.mean()
    avg_weekday_sales_per_day = weekday_sales.mean()

    # Define a threshold for significant differences
    threshold = 5000

    # Calculate the difference between the average sales
    sales_difference = avg_weekend_sales_per_day - avg_weekday_sales_per_day
    # Weekday vs Weekend Sales
    st.write(f"**Weekend Sales:** {format_currency(weekend_sales, 'INR', locale='en_IN').split('.')[0]}")
    st.write(f"**Weekday Sales:** {format_currency(weekday_sales, 'INR', locale='en_IN').split('.')[0]}")
    # Determine the observation based on threshold
    if abs(sales_difference) > threshold:
        if sales_difference > 0:
            st.write("**Observation:** Average sales on weekends are significantly higher than on weekdays.")
        else:
            st.write("**Observation:** Average sales on weekdays are significantly higher than on weekends.")
    else:
        st.write("**Observation:** Average sales on weekends and weekdays are not significantly different based on the threshold.")

    threshold_percentage = 0.05  # 5%
    total_sales = overall_trend.sum()
    threshold_value = total_sales * threshold_percentage

    st.header("Early vs Late Period Sales")

    # Determine midpoint based on data length
    mid_point = len(overall_trend) // 2
    first_period_sales = overall_trend.iloc[:mid_point].sum()
    second_period_sales = overall_trend.iloc[mid_point:].sum()

    st.write(f"**First Period Sales:** {format_currency(first_period_sales, 'INR', locale='en_IN').split('.')[0]}")
    st.write(f"**Second Period Sales:** {format_currency(second_period_sales, 'INR', locale='en_IN').split('.')[0]}")

    # Calculate the difference and compare with the threshold
    difference = np.abs(first_period_sales - second_period_sales)

    if difference > threshold_value:
        if first_period_sales > second_period_sales:
            st.write("**Observation:** Sales tend to be higher in the first part of the available data period.")
        else:
            st.write("**Observation:** Sales tend to be higher in the second part of the available data period.")
    else:
        st.write("**Observation:** The difference in sales between the periods is not significant enough to determine a clear trend.")
    # Specific Insights
    st.subheader("Specific Insights")

    # Highest and Lowest Sales Days
    # Determine the date corresponding to the highest and lowest sales
    highest_sales_day = overall_trend.idxmax()
    highest_sales_value = overall_trend.max()
    lowest_sales_day = overall_trend.idxmin()
    lowest_sales_value = overall_trend.min()

    # Map day number to actual dates
    highest_sales_date = filtered_data[filtered_data['Date'].dt.day == highest_sales_day]['Date'].iloc[0]
    lowest_sales_date = filtered_data[filtered_data['Date'].dt.day == lowest_sales_day]['Date'].iloc[0]

    # Format dates
    highest_sales_date_str = highest_sales_date.strftime('%Y-%m-%d')
    lowest_sales_date_str = lowest_sales_date.strftime('%Y-%m-%d')

    st.write(f"**Highest Sales Date:** {highest_sales_date_str} with sales of {format_currency(highest_sales_value, 'INR', locale='en_IN').split('.')[0]}")
    st.write(f"**Lowest Sales Date:** {lowest_sales_date_str} with sales of {format_currency(lowest_sales_value, 'INR', locale='en_IN').split('.')[0]}")

    # Define Indian events, holidays, long weekends, festivals, and shopping potential periods (January to July)
    indian_events = {
        'New Year': ['2024-01-01'],
        'Makar Sankranti': ['2024-01-14', '2024-01-15'],
        'Republic Day': ['2024-01-26'],
        'Valentine’s Week': ['2024-02-07', '2024-02-14'],
        'Maha Shivaratri': ['2024-03-08'],
        'Holi': ['2024-03-25', '2024-03-26'],
        'Good Friday': ['2024-03-29'],
        'Ramadan': ['2024-04-10', '2024-05-10'],  # Month-long period
        'Eid al-Fitr': ['2024-04-10'],
        
        'Summer Sales': ['2024-05-15', '2024-06-15'],  # Approx. month-long period
        'Long Weekend (June)': ['2024-06-29', '2024-06-30', '2024-07-01'],
        'Early Monsoon Sales': ['2024-07-15', '2024-07-30']  # July month-long period
    }

    # Convert event dates to datetime
    event_dates = [pd.to_datetime(date, format='%Y-%m-%d') for dates in indian_events.values() for date in dates]

    # Calculate event and non-event sales
    event_sales = filtered_data[filtered_data['Date'].isin(event_dates)]['Total'].sum()
    non_event_sales = filtered_data[~filtered_data['Date'].isin(event_dates)]['Total'].sum()
    # Filter data for event and non-event days
    event_sales_data = filtered_data[filtered_data['Date'].isin(event_dates)]
    non_event_sales_data = filtered_data[~filtered_data['Date'].isin(event_dates)]
    # Define a threshold for comparison (e.g., ₹1000)
    threshold = 10000
    average_event_sales = event_sales_data['Total'].mean()
    average_non_event_sales = non_event_sales_data['Total'].mean()
    num_event_days = event_sales_data['Date'].nunique()
    num_non_event_days = non_event_sales_data['Date'].nunique()

    # Display sales in INR format
    st.write(f"**Event Period Sales:** {format_currency(event_sales, 'INR', locale='en_IN').split('.')[0]} in {num_event_days} days")
    st.write(f"**Non-Event Period Sales:** {format_currency(non_event_sales, 'INR', locale='en_IN').split('.')[0]} in {num_non_event_days} days")

    if abs(average_event_sales - average_non_event_sales) <= threshold:
        st.write(f"**Observation:** Average sales during event and non-event periods are relatively similar ")
    elif average_event_sales > average_non_event_sales:
        st.write("**Observation:** Average sales during specific events tend to be higher.")
    else:
        st.write("**Observation:** Regular periods have higher average sales, indicating steady demand.")


    # Convert event dates to datetime
    event_dates = {event: [pd.to_datetime(date, format='%Y-%m-%d') for date in dates] for event, dates in indian_events.items()}

    # Calculate sales for each event
    event_sales = {}
    for event, dates in event_dates.items():
        event_sales[event] = filtered_data[filtered_data['Date'].isin(dates)]['Total'].sum()


    event_sales_df = pd.DataFrame(list(event_sales.items()), columns=['Event', 'Total Sales'])

    # Visualize sales for each event
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(event_sales_df['Event'], event_sales_df['Total Sales'], color='skyblue')
    ax.set_title('Total Sales for Each Event')
    ax.set_ylabel('Total Sales ')
    ax.set_xlabel('Event')
    # Set x-axis tick labels rotation and alignment
    ax.set_xticks(range(len(event_sales_df['Event'])))
    ax.set_xticklabels(event_sales_df['Event'], rotation=45, ha='right')

    # Add labels on bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 500, f"{format_currency(yval, 'INR', locale='en_IN').split('.')[0]}", ha='center', va='bottom')

    st.pyplot(fig)    

    # Display the plot in Streamlit
    st.header("Forecasted Revenue Trend ")
    # Generate the forecast data
    forecast_df = forecast_total_revenue(filtered_data, days=30)

    # Get the peak forecast date and value
    peak_forecast_date = forecast_df['ForecastedRevenue'].idxmax()
    max_forecast = forecast_df['ForecastedRevenue'].max()



    # Create the Plotly figure
    fig = go.Figure()

    # Add the forecasted revenue line
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['ForecastedRevenue'],
        mode='lines+markers',
        name='Forecasted Revenue',
        line=dict(dash='dash'),
        text=[f"{date}: ₹{value:,.2f}" for date, value in zip(forecast_df.index, forecast_df['ForecastedRevenue'])],
        hoverinfo='text'
    ))

    # Annotate the peak forecast
    fig.add_annotation(
        x=peak_forecast_date,
        y=max_forecast,
        text=f"Peak Forecast: ₹{max_forecast:,.2f}",
        showarrow=True,
        arrowhead=2
    )

    # Update layout for better readability
    fig.update_layout(
        title='Forecasted Revenue Trend',
        xaxis_title='Date',
        yaxis_title='Forecasted Revenue ',
        hovermode='closest'
    )


    st.plotly_chart(fig)

    # Most Profitable Products
    st.header("Top 10 Most Profitable Products")
    profit_data = filtered_data.groupby('ProductName').agg({'Profit': 'sum', 'Total': 'sum'}).reset_index()
    top_profitable_products = profit_data.sort_values(by='Profit', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=top_profitable_products, x='Profit', y='ProductName', ci=None, ax=ax)
    ax.set_title('Top 10 Most Profitable Products')
    ax.set_xlabel('Profit')
    ax.set_ylabel('Product')
    for p in ax.patches:
        if p.get_width() > 0:
            ax.annotate(f"₹{p.get_width():.0f}", (p.get_width(), p.get_y() + p.get_height() / 2), 
                        ha='left', va='center', xytext=(12, 0), textcoords='offset points')
    st.pyplot(fig)

    st.header("Sales Correlation with Price")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=filtered_data, x='Price_x', y='Total', hue='Category', ax=ax, legend=False)
    ax.set_title('Sales Correlation with Price')
    ax.set_xlabel('Price')
    ax.set_ylabel('Total Sales')
    st.pyplot(fig)
    correlation = filtered_data['Price_x'].corr(filtered_data['Total'])

    # Provide dynamic insights based on correlation
    if correlation > 0.5:
        st.write("**Positive Correlation Detected:**")
        st.write(f"- The analysis shows a strong positive correlation ({correlation:.2f}) between price and total sales. This suggests that higher-priced products tend to generate higher sales.")
        st.write("**Actions:**")
        st.write("  - **Premium Pricing Strategy:** Consider maintaining or increasing prices for high-demand products, as customers are willing to pay more.")
        st.write("  - **Focus on Value Proposition:** Highlight the unique value or quality of higher-priced products in marketing campaigns.")
        st.write("  - **Product Differentiation:** Explore opportunities to introduce premium versions of popular products to capitalize on this trend.")
    elif correlation < -0.5:
        st.write("**Negative Correlation Detected:**")
        st.write(f"- The analysis shows a strong negative correlation ({correlation:.2f}) between price and total sales. This indicates that lower-priced products tend to sell more.")
        st.write("**Actions:**")
        st.write("  - **Competitive Pricing:** Consider reducing prices for certain products to boost sales volume.")
        st.write("  - **Discount Strategies:** Implement targeted discount campaigns to attract price-sensitive customers.")
        st.write("  - **Inventory Management:** Monitor stock levels closely for lower-priced products to avoid stockouts due to high demand.")
    elif -0.5 <= correlation <= 0.5:
        st.write("**Weak or No Significant Correlation Detected:**")
        st.write(f"- The analysis shows a weak correlation ({correlation:.2f}) between price and total sales. This suggests that pricing might not be a key driver of sales in this case.")
        st.write("**Actions:**")
        st.write("  - **Customer Segmentation:** Focus on understanding different customer segments and their purchasing behavior, rather than relying on price adjustments alone.")
        st.write("  - **Enhanced Marketing:** Invest in marketing efforts that emphasize product features, benefits, and brand reputation rather than price.")
        st.write("  - **Price Testing:** Consider running A/B tests with different price points to identify any hidden pricing opportunities.")

    st.header("Association Rule Mining Insights")

    # Create a new column combining ProductName and Brand
    filtered_data['ProductBrand'] = filtered_data['ProductName'] + ' (' + filtered_data['Brand'] + ')'

    # Create a basket format where each row represents a transaction and each column represents a product with brand
    basket = filtered_data.groupby(['TransactionID', 'ProductBrand'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('TransactionID')

    # Convert quantities to boolean (0 or 1) to indicate presence or absence of a product in a transaction
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apply the Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1)

    # Check if there are any rules before proceeding
    if rules.empty:
        st.warning("No strong association found.")
    else:
        # Remove duplicates by ensuring only one direction of the rule is kept
        rules['sorted_sets'] = rules.apply(lambda x: frozenset(x['antecedents']).union(frozenset(x['consequents'])), axis=1)
        rules = rules.drop_duplicates(subset=['sorted_sets'])
        rules = rules.drop(columns=['sorted_sets'])

        # Convert frozenset to a readable string
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)).replace(', ', ''))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)).replace(', ', ''))

        # Display the top 10 unique rules
        st.write("Top 10 Unique Association Rules:")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        # Additional actionable insights
        st.write("""
        - **Product Pairing for Promotions**: Based on the association rules, products that frequently appear together in transactions can be bundled in promotions or placed close to each other in the store to increase sales.
        - **Cross-Selling Opportunities**: Use the antecedents and consequents from the rules to identify cross-selling opportunities. For example, if customers who buy Product A often buy Product B, these can be recommended together.
        - **Strategic Pricing**: Consider dynamic pricing strategies for products that appear in high-lift rules to maximize profitability without significantly affecting demand.
        - **Customer Targeting**: Use the frequent itemsets to create targeted marketing campaigns for customers who purchase specific combinations of products.
        """)


def customer_module():
   
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans
    from statsmodels.tsa.arima.model import ARIMA
    import plotly.express as px

    # Load Data
    @st.cache_data
    def load_data():
        customer_df = pd.read_csv('customer_data.csv')  # Replace with actual file paths
        sales_df = pd.read_csv('sales_data.csv')
        products_df = pd.read_csv('products_data.csv')
        return customer_df, sales_df, products_df

    customer_df, sales_df, products_df = load_data()

    # Title
    st.title("Customer Insights Dashboard")

    # Univariate Analysis with Insights
    st.header("Customer Demographics Overview")

        # Age Distribution with Insights
    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(customer_df['Age'], kde=True, ax=ax)
    st.pyplot(fig)

    age_median = customer_df['Age'].median()
    st.markdown(f"""
    - **Insight:** The majority of customers fall within the 25-40 age range with a median age of {age_median}. 
    - **Action:** Tailor marketing efforts towards this demographic.
    """)

    # Gender Distribution with Insights
    st.subheader("Gender Distribution")
    gender_counts = customer_df['Gender'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['#66b3ff','#ff9999'], startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown(f"""
    - **Insight:** Gender distribution is relatively balanced, with {gender_counts['Male']} males and {gender_counts['Female']} females.
    - **Action:** Ensure product variety and marketing campaigns appeal to both genders.
    """)

    # Feature selection for clustering
    features = customer_df[['Age', 'Gender', 'Location', 'LoyaltyScore']]
    column_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(), ['Gender', 'Location']),
        ('scaler', StandardScaler(), ['Age', 'LoyaltyScore'])
    ])
    processed_features = column_transformer.fit_transform(features)

    # Determine optimal number of clusters
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(processed_features)
        wcss.append(kmeans.inertia_)



    # Fit the KMeans algorithm with the optimal number of clusters
    optimal_clusters = 4  # Assume we determined 4 is the optimal number from the elbow method
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    customer_df['Cluster'] = kmeans.fit_predict(processed_features)

    # Clustered Customer Segmentation
    st.header("Customer Segmentation")
    st.subheader("Segmentation Based on Age and Loyalty Score")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(data=customer_df, x='Age', y='LoyaltyScore', hue='Cluster', palette='viridis', s=100, ax=ax)
    st.pyplot(fig)

    # Dynamic Recommendations Based on Clusters with Visualizations
    st.markdown("### Cluster-Based Insights and Actions")
    for cluster in range(optimal_clusters):
        cluster_data = customer_df[customer_df['Cluster'] == cluster]
        age_median = cluster_data['Age'].median()
        loyalty_median = cluster_data['LoyaltyScore'].median()
        location_mode = cluster_data['Location'].mode()[0]
        
        with st.expander(f"Cluster {cluster + 1} Insights and Actions", expanded=False):
            st.markdown(f"""
            - **Age:** The median age in this cluster is {age_median}.
            - **Loyalty Score:** The median loyalty score is {loyalty_median}.
            - **Location:** The most common location for this cluster is {location_mode}.
            - **Action:** Focus marketing efforts on customers in {location_mode} and consider loyalty programs to enhance the loyalty score further.
            """)

            # Visualization: Age Distribution in Cluster
            st.subheader(f"Age Distribution in Cluster {cluster + 1}")
            fig, ax = plt.subplots()
            sns.histplot(cluster_data['Age'], kde=True, ax=ax)
            ax.set_title(f'Age Distribution in Cluster {cluster + 1}')
            st.pyplot(fig)

            # Visualization: Loyalty Score Distribution in Cluster
            st.subheader(f"Loyalty Score Distribution in Cluster {cluster + 1}")
            fig, ax = plt.subplots()
            sns.histplot(cluster_data['LoyaltyScore'], kde=True, ax=ax)
            ax.set_title(f'Loyalty Score Distribution in Cluster {cluster + 1}')
            st.pyplot(fig)

            # Visualization: Location Distribution in Cluster
            st.subheader(f"Location Distribution in Cluster {cluster + 1}")
            fig, ax = plt.subplots()
            location_counts = cluster_data['Location'].value_counts()
            ax.pie(location_counts, labels=location_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            ax.set_title(f'Location Distribution in Cluster {cluster + 1}')
            st.pyplot(fig)

    # Product Analysis per Cluster
    st.header("Product Preferences by Cluster")
    selected_cluster = st.selectbox("Select Cluster", customer_df['Cluster'].unique())

    cluster_sales_data = sales_df.merge(customer_df[['CustomerID', 'Cluster']], on='CustomerID')
    cluster_product_summary = cluster_sales_data[cluster_sales_data['Cluster'] == selected_cluster].groupby('ProductID').agg({'Quantity': 'sum', 'Total': 'sum'}).reset_index()
    cluster_product_summary = cluster_product_summary.merge(products_df[['ProductID', 'ProductName']], on='ProductID')

    top_products_by_quantity = cluster_product_summary.sort_values('Quantity', ascending=False).head(5)

    fig, ax = plt.subplots()
    sns.barplot(x='Quantity', y='ProductName', data=top_products_by_quantity, ax=ax)
    st.pyplot(fig)

    st.markdown(f"""
    - **Insight:** The top-selling products in Cluster {selected_cluster} are displayed above.
    - **Action:** Ensure stock availability for these products, consider bundling them with other items, or run targeted promotions.
    """)

    # Define the forecast_cluster_revenue function
    def forecast_cluster_revenue(cluster_id, data, days=30):
        cluster_data = data[data['Cluster'] == cluster_id]
        daily_revenue = cluster_data.groupby('Date').agg({'Total': 'sum'}).reset_index()
        daily_revenue['Date'] = pd.to_datetime(daily_revenue['Date'])
        daily_revenue.set_index('Date', inplace=True)
        
        model_revenue = ARIMA(daily_revenue['Total'], order=(5, 1, 0))
        model_revenue_fit = model_revenue.fit()
        forecast_revenue = model_revenue_fit.forecast(steps=days)
        
        forecast_dates = pd.date_range(start=daily_revenue.index[-1] + pd.Timedelta(days=1), periods=days)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'ForecastedRevenue': forecast_revenue})
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df

    # Revenue Share of Clusters and Forecast
    st.header("Revenue Share Analysis")
    # Revenue share of clusters
    cluster_revenue = cluster_sales_data.groupby('Cluster').agg({'Total': 'sum'}).reset_index()
    fig = px.pie(cluster_revenue, values='Total', names='Cluster', title='Revenue Share by Cluster')
    st.plotly_chart(fig)

    # Forecasted revenue share
    st.subheader(f"Revenue Forecast for Cluster {selected_cluster} - Next 30 Days")
    forecast_df = forecast_cluster_revenue(selected_cluster, cluster_sales_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(forecast_df.index, forecast_df['ForecastedRevenue'], marker='o')
    ax.set_title(f'Forecasted Revenue for Cluster {selected_cluster} for Next 30 Days')
    ax.set_xlabel('Date')
    ax.set_ylabel('Forecasted Revenue')
    st.pyplot(fig)

    st.markdown(f"""
    - **Insight:** The revenue forecast for Cluster {selected_cluster} over the next 30 days helps anticipate future sales trends.
    - **Action:** Adjust inventory levels, marketing efforts, and staffing based on expected demand for this cluster.
    """)


    # Final Dynamic Recommendations
    st.header("Actionable Recommendations")
    for cluster in range(optimal_clusters):
        cluster_data = customer_df[customer_df['Cluster'] == cluster]
        top_products = cluster_sales_data[cluster_sales_data['Cluster'] == cluster].groupby('ProductID').agg({'Quantity': 'sum'}).reset_index()
        top_products = top_products.merge(products_df[['ProductID', 'ProductName']], on='ProductID').sort_values('Quantity', ascending=False).head(3)
        
        st.subheader(f"Cluster {cluster + 1} Recommendations")
        st.markdown(f"""
        1. **Top Products:** Focus on maintaining stock for top products like {', '.join(top_products['ProductName'].values)}.
        2. **Targeted Promotions:** Use demographic and behavioral insights to create personalized marketing campaigns.
        3. **Loyalty Programs:** Consider implementing loyalty programs for this cluster to improve customer retention.
        4. **Revenue Optimization:** Utilize revenue forecasts to make informed decisions on pricing, promotions, and inventory adjustments.
        """)




# Run the application
if __name__ == "__main__":
    run_app()
