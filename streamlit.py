import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from fpdf import FPDF # Library for PDF generation

# Set Streamlit page configuration
st.set_page_config(layout="wide", page_title="Gemini Marketing Dashboard")

# Custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #333;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 9999px; /* Tailwind rounded-full */
            padding: 10px 20px;
            font-weight: bold;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* Tailwind shadow-lg */
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); /* Tailwind shadow-xl */
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .stMarkdown h1 {
            color: #4338CA; /* Indigo-700 */
            text-align: center;
            font-size: 2.5rem; /* text-4xl */
            font-weight: 700; /* font-bold */
            margin-bottom: 2rem;
            border-radius: 0.5rem; /* rounded-lg */
            padding: 0.5rem;
        }
        .stMarkdown h2 {
            color: #312E81; /* Indigo-800 or Blue-800 or Purple-800 */
            font-size: 1.5rem; /* text-2xl */
            font-weight: 600; /* font-semibold */
            margin-bottom: 1rem;
        }
        .stMarkdown h3 {
            color: #1F2937; /* Gray-900 */
            font-size: 1.25rem; /* text-xl */
            font-weight: 500; /* font-medium */
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .stMarkdown h4 {
            color: #374151; /* Gray-800 */
            font-size: 1rem; /* text-md */
            font-weight: 600; /* font-semibold */
            margin-bottom: 0.5rem;
        }
        .stMarkdown ul {
            list-style-type: disc;
            margin-left: 1.25rem;
            color: #4B5563; /* Gray-700 */
        }
        .stMarkdown li {
            margin-bottom: 0.25rem;
        }
        .stAlert {
            border-radius: 0.5rem;
        }
        .stFileUploader {
            border-radius: 0.5rem;
        }
        .stSelectbox>div>div {
            border-radius: 0.375rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .stDateInput>div>div {
            border-radius: 0.375rem;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        }
        .container-bg-indigo-50 {
            background-color: #EEF2FF; /* indigo-50 */
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06); /* shadow-inner */
            margin-bottom: 2rem;
        }
        .container-bg-blue-50 {
            background-color: #EFF6FF; /* blue-50 */
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 2rem;
        }
        .container-bg-purple-50 {
            background-color: #F5F3FF; /* purple-50 */
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
            margin-bottom: 2rem;
        }
        .chart-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06); /* shadow-md */
            transition: box-shadow 0.3s ease-in-out;
            margin-bottom: 1rem;
        }
        .chart-card:hover {
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
        }
    </style>
""", unsafe_allow_html=True)

# Function to clean data
def clean_data(df):
    # Convert 'Date' to datetime, coerce errors to NaT
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # Fill empty 'Engagements' with 0, convert to int
    df['Engagements'] = pd.to_numeric(df['Engagements'], errors='coerce').fillna(0).astype(int)

    # Fill missing categorical data with 'Unknown' or 'Other'
    df['Platform'] = df['Platform'].fillna('Unknown')
    df['Sentiment'] = df['Sentiment'].fillna('Neutral')
    df['Media Type'] = df['Media Type'].fillna('Other')
    df['Location'] = df['Location'].fillna('Unknown')

    return df

# Static insights for each chart
def get_insights(chart_name):
    insights = []
    if chart_name == 'sentiment':
        insights = [
            '1. Identify the dominant sentiment: A large slice of positive sentiment indicates successful content.',
            '2. Analyze negative sentiment: A significant negative slice suggests areas for improvement in content or product.',
            '3. Monitor neutral sentiment: A high neutral percentage might mean content isn\'t strongly resonating, providing an opportunity to refine messaging.',
        ]
    elif chart_name == 'engagement_trend':
        insights = [
            '1. Pinpoint engagement peaks: Spikes indicate highly successful campaigns or content releases.',
            '2. Identify engagement troughs: Drops suggest content fatigue or less effective campaigns during those periods.',
            '3. Observe seasonality: Recurring patterns might reveal weekly, monthly, or yearly trends that can inform future scheduling.',
        ]
    elif chart_name == 'platform':
        insights = [
            '1. Discover top-performing platforms: Focus resources where engagement is highest.',
            '2. Identify underperforming platforms: Evaluate content strategy or audience targeting for these platforms.',
            '3. Spot platform-specific trends: Different platforms may show unique engagement patterns, requiring tailored content.',
        ]
    elif chart_name == 'media_type':
        insights = [
            '1. Determine preferred media formats: Invest more in media types that yield higher engagement.',
            '2. Diversify content strategy: If one media type dominates, consider experimenting with others to reach new audiences.',
            '3. Assess cost-effectiveness: Compare engagement of different media types against their production costs.',
        ]
    elif chart_name == 'locations':
        insights = [
            '1. Identify key geographical markets: Focus marketing efforts on locations with high engagement.',
            '2. Uncover untapped markets: Locations with lower engagement might represent opportunities for targeted campaigns.',
            '3. Understand regional preferences: Different locations may respond better to specific content themes or languages.',
        ]
    return "\n".join([f"- {i}" for i in insights])

# --- Streamlit App Layout ---

st.title("Gemini Marketing Dashboard")

# File Upload Section
st.markdown('<div class="container-bg-indigo-50">', unsafe_allow_html=True)
st.markdown('<h2>Upload CSV Data</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your CSV file here", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = clean_data(df.copy()) # Clean a copy of the dataframe
        st.success("CSV loaded and processed successfully!")
    except Exception as e:
        st.error(f"Error loading or processing CSV: {e}")

if df is not None and not df.empty:
    # --- Filters Section ---
    st.markdown('<div class="container-bg-blue-50">', unsafe_allow_html=True)
    st.markdown('<h2>Filter Data</h2>', unsafe_allow_html=True)

    # Get unique values for filters
    all_platforms = ['All'] + sorted(df['Platform'].unique().tolist())
    all_sentiments = ['All'] + sorted(df['Sentiment'].unique().tolist())
    all_media_types = ['All'] + sorted(df['Media Type'].unique().tolist())
    all_locations = ['All'] + sorted(df['Location'].unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_platform = st.selectbox("Platform", all_platforms)
    with col2:
        selected_sentiment = st.selectbox("Sentiment", all_sentiments)
    with col3:
        selected_media_type = st.selectbox("Media Type", all_media_types)

    col4, col5 = st.columns(2)
    with col4:
        selected_location = st.selectbox("Location", all_locations)
    with col5:
        # Date range filter
        min_date = df['Date'].min() if not df['Date'].isnull().all() else pd.to_datetime('2020-01-01')
        max_date = df['Date'].max() if not df['Date'].isnull().all() else pd.to_datetime('2025-12-31')

        # Ensure min_date and max_date are not NaT
        if pd.isna(min_date):
            min_date = pd.to_datetime('2020-01-01')
        if pd.isna(max_date):
            max_date = pd.to_datetime('2025-12-31')

        try:
            date_range = st.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                # Ensure end_date includes the entire day
                end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            else:
                start_date = min_date
                end_date = max_date
        except Exception as e:
            st.warning(f"Could not parse date range, showing all dates: {e}")
            start_date = min_date
            end_date = max_date


    # Apply filters
    filtered_df = df.copy()
    if selected_platform != 'All':
        filtered_df = filtered_df[filtered_df['Platform'] == selected_platform]
    if selected_sentiment != 'All':
        filtered_df = filtered_df[filtered_df['Sentiment'] == selected_sentiment]
    if selected_media_type != 'All':
        filtered_df = filtered_df[filtered_df['Media Type'] == selected_media_type]
    if selected_location != 'All':
        filtered_df = filtered_df[filtered_df['Location'] == selected_location]

    # Date filtering
    filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)]

    st.markdown('</div>', unsafe_allow_html=True) # Close filters div

    # --- Dashboard Content ---
    st.markdown('<div class="chart-section">', unsafe_allow_html=True) # Wrapper for charts

    st.markdown('<h2>Dashboard Overview</h2>', unsafe_allow_html=True)

    if filtered_df.empty:
        st.warning("No data matches the current filter criteria. Try adjusting your filters.")
    else:
        # Create a list to store plot images for PDF export
        plot_images = []

        # 1. Sentiment Breakdown (Pie Chart)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<h3>Sentiment Breakdown</h3>', unsafe_allow_html=True)
        sentiment_counts = filtered_df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment',
                                title='Sentiment Breakdown', hole=0.4,
                                color_discrete_map={'Positive':'#4CAF50', 'Neutral':'#FFC107', 'Negative':'#F44336'})
        fig_sentiment.update_traces(textinfo='percent+label', pull=[0.05 if s == 'Positive' else 0 for s in sentiment_counts['Sentiment']])
        fig_sentiment.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), showlegend=True)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        st.markdown('<h4>Key Insights:</h4>')
        st.markdown(get_insights('sentiment'), unsafe_allow_html=True)
        plot_images.append(fig_sentiment.to_image(format="png"))
        st.markdown('</div>', unsafe_allow_html=True)


        # 2. Engagement Trend over Time (Line Chart)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<h3>Engagement Trend over Time</h3>', unsafe_allow_html=True)
        engagement_trend = filtered_df.groupby(filtered_df['Date'].dt.date)['Engagements'].sum().reset_index()
        engagement_trend.columns = ['Date', 'Total Engagements']
        fig_engagement_trend = px.line(engagement_trend, x='Date', y='Total Engagements',
                                        title='Engagement Trend over Time')
        fig_engagement_trend.update_traces(mode='lines+markers', line=dict(color='#2196F3', width=2), marker=dict(size=6))
        fig_engagement_trend.update_layout(height=350, margin=dict(t=50, b=80, l=60, r=20))
        st.plotly_chart(fig_engagement_trend, use_container_width=True)
        st.markdown('<h4>Key Insights:</h4>')
        st.markdown(get_insights('engagement_trend'), unsafe_allow_html=True)
        plot_images.append(fig_engagement_trend.to_image(format="png"))
        st.markdown('</div>', unsafe_allow_html=True)


        # 3. Platform Engagements (Bar Chart)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<h3>Platform Engagements</h3>', unsafe_allow_html=True)
        platform_engagements = filtered_df.groupby('Platform')['Engagements'].sum().reset_index()
        platform_engagements = platform_engagements.sort_values(by='Engagements', ascending=False)
        fig_platform = px.bar(platform_engagements, x='Platform', y='Engagements',
                               title='Platform Engagements', color_discrete_sequence=['#FF9800'])
        fig_platform.update_layout(height=350, margin=dict(t=50, b=60, l=60, r=20))
        st.plotly_chart(fig_platform, use_container_width=True)
        st.markdown('<h4>Key Insights:</h4>')
        st.markdown(get_insights('platform'), unsafe_allow_html=True)
        plot_images.append(fig_platform.to_image(format="png"))
        st.markdown('</div>', unsafe_allow_html=True)


        # 4. Media Type Mix (Pie Chart)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<h3>Media Type Mix</h3>', unsafe_allow_html=True)
        media_type_counts = filtered_df['Media Type'].value_counts().reset_index()
        media_type_counts.columns = ['Media Type', 'Count']
        fig_media_type = px.pie(media_type_counts, values='Count', names='Media Type',
                                title='Media Type Mix', hole=0.4,
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_media_type.update_traces(textinfo='percent+label')
        fig_media_type.update_layout(height=350, margin=dict(t=50, b=20, l=20, r=20), showlegend=True)
        st.plotly_chart(fig_media_type, use_container_width=True)
        st.markdown('<h4>Key Insights:</h4>')
        st.markdown(get_insights('media_type'), unsafe_allow_html=True)
        plot_images.append(fig_media_type.to_image(format="png"))
        st.markdown('</div>', unsafe_allow_html=True)


        # 5. Top 5 Locations (Bar Chart)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<h3>Top 5 Locations by Engagements</h3>', unsafe_allow_html=True)
        location_engagements = filtered_df.groupby('Location')['Engagements'].sum().reset_index()
        top_5_locations = location_engagements.sort_values(by='Engagements', ascending=False).head(5)
        fig_locations = px.bar(top_5_locations, x='Location', y='Engagements',
                                title='Top 5 Locations by Engagements', color_discrete_sequence=['#673AB7'])
        fig_locations.update_layout(height=350, margin=dict(t=50, b=60, l=60, r=20))
        st.plotly_chart(fig_locations, use_container_width=True)
        st.markdown('<h4>Key Insights:</h4>')
        st.markdown(get_insights('locations'), unsafe_allow_html=True)
        plot_images.append(fig_locations.to_image(format="png"))
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # Close charts section wrapper

    # --- Campaign Strategy Summary ---
    st.markdown('<div class="container-bg-purple-50">', unsafe_allow_html=True)
    st.markdown('<h2>Campaign Strategy Summary</h2>', unsafe_allow_html=True)
    campaign_strategy_summary = """
        Based on the current data, a successful campaign strategy should prioritize content that drives high engagement on platforms like **[Top Platform]**.
        Focus on creating more **[Top Media Type]** content, as it consistently performs well.
        Addressing negative sentiment through **proactive customer service and content refinement** and reinforcing positive engagement with **interactive and trending content** will be crucial.
        Consider targeted campaigns in **[Top Location 1]** and **[Top Location 2]** to capitalize on strong regional interest.
        Future campaigns should also leverage insights from engagement trends to schedule content during **peak periods** (e.g., specific days of the week or times of day identified from the trend chart).
        """
    st.markdown(campaign_strategy_summary, unsafe_allow_html=True)
    st.markdown("""
    <p class="text-sm text-gray-500 italic mt-4">
        Note: In a real-world scenario, this summary could be dynamically generated by a large language model
        based on the analyzed data and insights.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Export PDF Button ---
    @st.cache_data
    def create_pdf(plot_images_data):
        pdf = FPDF(format='A4')
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Gemini Marketing Dashboard", ln=True, align='C')
        pdf.set_font("Arial", size=10)
        pdf.ln(10)

        # Add strategy summary
        pdf.multi_cell(0, 5, txt="Campaign Strategy Summary:", align='L')
        pdf.ln(2)
        pdf.multi_cell(0, 5, txt=campaign_strategy_summary.replace("**", ""), align='L') # Remove markdown bold for PDF
        pdf.ln(10)

        # Add images of plots
        for img_data in plot_images_data:
            try:
                # Convert bytes to base64 for embedding
                img_base64 = base64.b64encode(img_data).decode()
                # Determine image type for FPDF
                if img_data.startswith(b'\x89PNG\r\n\x1a\n'): # PNG magic number
                    img_type = 'PNG'
                elif img_data.startswith(b'\xFF\xD8\xFF'): # JPEG magic number
                    img_type = 'JPEG'
                else:
                    img_type = 'PNG' # Default or try to infer

                # Add image to PDF. Adjust width to fit A4.
                # A4 width is 210mm. Assuming image width is around 700px, scale it.
                # 1 mm = 3.7795275591 px
                # So if image is 700px, in mm it's 700 / 3.7795 = 185mm approx.
                # A4 width in mm is 210mm, so 185mm fits with some margin.
                # We'll use 190mm width, centered.
                pdf.image(BytesIO(img_data), x=10, w=190)
                pdf.ln(5) # Small line break after each image
            except Exception as e:
                st.error(f"Error adding image to PDF: {e}")
                pdf.multi_cell(0, 5, txt=f"Error loading chart image: {e}", align='C')
                pdf.ln(5)
            # Add a new page if the next content won't fit
            if pdf.get_y() > (pdf.h - 40): # If less than 40mm space left
                pdf.add_page()

        return pdf.output(dest='S').encode('latin1') # Return as bytes

    if st.button("Export Dashboard as PDF"):
        if plot_images:
            with st.spinner("Generating PDF..."):
                pdf_output = create_pdf(plot_images)
                st.download_button(
                    label="Download PDF",
                    data=pdf_output,
                    file_name="gemini_marketing_dashboard.pdf",
                    mime="application/pdf"
                )
                st.success("PDF generated successfully!")
        else:
            st.warning("No charts to export. Please upload data and apply filters first.")

else:
    st.info("Please upload a CSV file to begin.")

