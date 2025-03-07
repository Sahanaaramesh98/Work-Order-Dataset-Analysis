import pandas as pd
import google as google
import json
import os
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from scipy.stats import zscore
import streamlit as st
import time

# SETTING UP LLM
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

# Safety settings
safe = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro", 
    generation_config=generation_config, 
    safety_settings=safe
)

chat_session = model.start_chat(history=[])

# LOADING DATA (Update the path to your CSV file)
new_df = pd.read_csv(r"C:\Users\HP\OneDrive - Manipal Academy of Higher Education\Desktop\Certifications\Data analytics\work-order-management-module.csv")
# Remove rows with missing values in workorder_activity_code since it contains Client information which can be filled with assumptions
cleaning_df = new_df.dropna(subset=['WORKORDER_ACTIVITY_CODE'])
# Remove duplicates rows
cleaning_df = cleaning_df.drop_duplicates()

cleaning_df['WORKORDER_STARTED'] = pd.to_datetime(cleaning_df['WORKORDER_STARTED'], errors='coerce')
cleaning_df['WORKORDER_COMPLETED'] = pd.to_datetime(cleaning_df['WORKORDER_COMPLETED'], errors='coerce')
cleaning_df['WORKORDER_ADDED'] = pd.to_datetime(cleaning_df['WORKORDER_ADDED'], errors='coerce')

# Calculate the difference in seconds between workorder_started and workorder_completed to understand the relationship between columns
# We consider the smallest unit seconds instead of any other unit
cleaning_df['sec_btw_startAndadd'] = (cleaning_df['WORKORDER_ADDED'] - cleaning_df['WORKORDER_STARTED']).dt.total_seconds()
cleaning_df['dateTime_btw_startAndadd'] = pd.to_timedelta(cleaning_df['sec_btw_startAndadd'], unit='s')

# Remove noisy data with negative seconds from dataframe
cleaning_df = cleaning_df[cleaning_df['sec_btw_startAndadd'] > 0].reset_index(drop=True)

# Calculate the difference in seconds between workorder_started and workorder_completed to understand the relationship between columns
# We consider the smallest unit seconds instead of any other unit
cleaning_df['time_taken_activity'] = pd.to_timedelta(cleaning_df['WORKORDER_COMPLETED'] - cleaning_df['WORKORDER_STARTED'])

cleaning_df = cleaning_df.drop(index=cleaning_df[cleaning_df['WORKORDER_COMPLETED'] <= cleaning_df['WORKORDER_STARTED']].index).reset_index(drop=True)

# Convert timedelta to total seconds as float for calculation
cleaning_df['time_taken_activity_seconds'] = cleaning_df['time_taken_activity'].dt.total_seconds()

lower_bound = cleaning_df['time_taken_activity_seconds'].quantile(0.1)
upper_bound = cleaning_df['time_taken_activity_seconds'].quantile(0.99)
cleaning_df = cleaning_df[(cleaning_df['time_taken_activity_seconds'] >= lower_bound) & (cleaning_df['time_taken_activity_seconds'] <= upper_bound)  | cleaning_df['time_taken_activity_seconds'].isna()].reset_index(drop=True)

cleaning_df['time_taken_activity'] = cleaning_df['time_taken_activity'].fillna(cleaning_df.groupby('WORKORDER_ACTIVITY_DESCRIPTION')['time_taken_activity'].transform('mean'))
cleaning_df['time_taken_activity_seconds'] = cleaning_df['time_taken_activity_seconds'].fillna(cleaning_df.groupby('WORKORDER_ACTIVITY_DESCRIPTION')['time_taken_activity_seconds'].transform('mean'))

cleaning_df = cleaning_df.dropna(subset=['time_taken_activity'])
new_df = cleaning_df.reset_index(drop=True)

# Create a prompt for Gemini
prompt = """
I have a dataset containing work orders. The column 'workorder_activity_description' represents the type of task performed.
Analyze the dataset and summarize:
- The most frequent work order descriptions
- The tasks that take the most time
- The average time taken for each work order

Here is a sample of my data:
{}
""".format(new_df.head().to_string())

# Get response from Gemini
response = model.generate_content(prompt)

# Print Gemini's response
print(response.text)

# SQL Query generation
sql_prompt = """
I have a table called 'work_orders' with the following columns:
- workorder_activity_description (text)
- time_taken_activity_seconds (integer)
- workorder_activity_code (text)

Write an SQL query to find the top 5 most frequent work orders along with their average time taken.
"""

# Get SQL response from Gemini
response_sql = model.generate_content(sql_prompt)

# Print the SQL Query
print("Generated SQL Query:\n", response_sql.text)

# Streamlit UI
st.title("Work Order Analysis Chatbot")
st.write("Ask me anything about your work orders!")

# User input
user_question = st.text_input("Enter your question:")

if st.button("Ask Gemini"):
    if user_question:
        if "most frequent work order descriptions" in user_question.lower():
            # Calculate most frequent descriptions
            description_counts = new_df['WORKORDER_ACTIVITY_DESCRIPTION'].value_counts()
            top_descriptions = description_counts.head(10)  # Show top 10 most frequent descriptions
            
            st.write("**Top 10 Most Frequent Work Order Descriptions:**")
            st.write(top_descriptions)

        elif "frequent tasks for a particular client" in user_question.lower():
            recent_clients = new_df.sort_values(by="workorder_started", ascending=False)\
                .drop_duplicates(subset=['workorder_activity_code'])[['workorder_activity_code', 'workorder_started']]\
                .head(10).reset_index(drop=True)
            st.write("**Frequent Tasks for Recent Clients:**")
            st.dataframe(recent_clients)

        elif "clients with similar types of issues" in user_question.lower():
            client_grouped = new_df.groupby('workorder_activity_description')['workorder_activity_code'].nunique().reset_index()
            client_grouped.columns = ['Work Order Description', 'Unique Issue Types']
            st.write("**Clients with Similar Types of Issues:**")
            st.dataframe(client_grouped)

        elif "average time required for each task" in user_question.lower():
            avg_seconds = new_df.groupby('workorder_activity_description')['time_taken_activity_seconds'].mean().reset_index()
            avg_seconds['avg_time_taken_activity'] = pd.to_timedelta(avg_seconds['time_taken_activity_seconds'], unit='s')
            st.write("**Average Time Required for Each Task:**")
            st.dataframe(avg_seconds[['workorder_activity_description', 'avg_time_taken_activity']])

        elif "task that takes maximum time" in user_question.lower():
            max_time = new_df['time_taken_activity_seconds'].max()
            max_time_task = new_df[new_df['time_taken_activity_seconds'] == max_time]['workorder_activity_description'].iloc[0]
            avg_time_for_max_task = new_df[new_df['workorder_activity_description'] == max_time_task]['time_taken_activity_seconds'].mean()

            st.markdown(f"**Task with Maximum Time:** {max_time_task}")
            st.markdown(f"**Maximum Time Taken:** {pd.to_timedelta(max_time, unit='s')}")
            st.markdown(f"**Average Time for this Task:** {pd.to_timedelta(avg_time_for_max_task, unit='s')}")

        elif "ranges of time per task" in user_question.lower():
            min_seconds = new_df.groupby('workorder_activity_description')['time_taken_activity_seconds'].min().reset_index()
            max_seconds = new_df.groupby('workorder_activity_description')['time_taken_activity_seconds'].max().reset_index()

            min_seconds['min_time_taken_activity'] = pd.to_timedelta(min_seconds['time_taken_activity_seconds'], unit='s')
            max_seconds['max_time_taken_activity'] = pd.to_timedelta(max_seconds['time_taken_activity_seconds'], unit='s')

            time_range = pd.merge(min_seconds, max_seconds, on='workorder_activity_description', how='inner')
            st.write("**Time Range per Task:**")
            st.dataframe(time_range[['workorder_activity_description', 'min_time_taken_activity', 'max_time_taken_activity']])

        else:
            # If question doesn't match predefined ones, send to Gemini
            prompt = f"""
            Here is a dataset containing work orders.
            Columns: workorder_activity_description, time_taken_activity_seconds, workorder_activity_code.
            Answer the following question based on this dataset:
            {user_question}
            Here is a sample of the dataset:
            {new_df.head().to_string()}
            """
            
            # Call Gemini API with the prompt
            response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
            
            # Display the response in the Streamlit app
            st.write("**Response:**", response.text)
    else:
        st.write("Please enter a question.")

        import time

try:
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)
except google.api_core.exceptions.ResourceExhausted:
    st.warning("Quota exceeded! Retrying in 10 seconds...")
    time.sleep(10)  # Wait before retrying
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content(prompt)



