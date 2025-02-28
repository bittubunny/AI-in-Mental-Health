import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import sqlite3
import time  # Import time module
import numpy as np
# Load the trained model
model = joblib.load('voting_gb_dt_model.pkl')

# Connect to SQLite database
conn = sqlite3.connect('new_user_data.db')
c = conn.cursor()

# Create tables if they don't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        password TEXT
    )
''')

# Create a new predictions table with the time_spent column if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS predictions_new (
        username TEXT,
        date DATETIME,
        prediction INTEGER,
        status TEXT,
        time_spent INTEGER,  -- New column for time spent in seconds
        FOREIGN KEY (username) REFERENCES users (username)
    )
''')
# Create a new chat messages table if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

conn.commit()

# Check if the old predictions table exists and migrate data if necessary
c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
if c.fetchone() is not None:
    # Copy data from the old predictions table to the new one
    c.execute('''
        INSERT INTO predictions_new (username, date, prediction, status)
        SELECT username, date, prediction, status FROM predictions
    ''')
    conn.commit()
    
    # Drop the old predictions table
    c.execute('DROP TABLE predictions')
    conn.commit()

# Rename the new table to the original table name
c.execute('ALTER TABLE predictions_new RENAME TO predictions')
conn.commit()

# Create admin account if it doesn't exist
def create_admin_account():
    admin_username = "admin"
    admin_password = "admin891"  # Change this to a more secure password
    c.execute("INSERT OR IGNORE INTO users (username, password) VALUES (?, ?)", (admin_username, admin_password))
    conn.commit()

create_admin_account()

# Function to authenticate user
def authenticate(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return c.fetchone() is not None

# Function to register user
def register(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# Function to save prediction to the database
def save_prediction(username, date, prediction, status, time_spent):
    c.execute("INSERT INTO predictions (username, date, prediction, status, time_spent) VALUES (?, ?, ?, ?, ?)", 
              (username, date, prediction, status, time_spent))
    conn.commit()

# Function to fetch predictions for a user
def fetch_predictions(username):
    c.execute("SELECT date, prediction, status, time_spent FROM predictions WHERE username=?", (username,))
    rows = c.fetchall()
    return pd.DataFrame(rows, columns=["Date", "Prediction", "Status", "Time Spent"])

# Function to fetch chat messages with IDs
def fetch_chat_messages():
    c.execute("SELECT id, username, message, timestamp FROM chat_messages ORDER BY timestamp")
    return c.fetchall()

# Function to save a chat message
def save_chat_message(username, message):
    c.execute("INSERT INTO chat_messages (username, message) VALUES (?, ?)", (username, message))
    conn.commit()

# Function to delete a chat message
def delete_chat_message(message_id):
    c.execute("DELETE FROM chat_messages WHERE id=?", (message_id,))
    conn.commit()

# Function to make predictions and map to mental health status
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df)  # This returns the probabilities for each class
    confidence_percentage = np.max(confidence) * 100  # Assuming binary classification

    return prediction[0], confidence_percentage

def map_to_status(yes_count):
    if yes_count <= 3:
        return "Stable or Low Instability"
    elif yes_count == 4:
        return "Moderate Instability"
    elif 5 <= yes_count <= 8:
        return "High Instability or Severe Instability"

# Function to update admin password
def update_admin_password(new_password):
    c.execute("UPDATE users SET password=? WHERE username='admin'", (new_password,))
    conn.commit()

# Initialize session state for authentication and mood tracking
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.login_time = None  # Track login time

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

if "predictions" not in st.session_state:
    st.session_state.predictions = pd.DataFrame(columns=["Date", "Prediction", "Status", "Time Spent"])

# Navigation
st.markdown("<h1 style='text-align: left; color:rgb(0, 1, 75);'>🤖 AI in Mental Health: Detecting Early Signs of Instability 🧠</h1>", unsafe_allow_html=True)

page = st.sidebar.selectbox("Select Page", ["Home", "Mood Tracking", "Personalized Recommendations", "Admin Dashboard", "Connect Page"])


if page == "Home":
    if not st.session_state.logged_in:
        option = st.radio("Login or Register", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if option == "Register":
            if st.button("Register"):
                if username.strip() == "":
                    st.error("Username cannot be empty!")
                elif register(username, password):
                    st.success("Registered successfully! Please log in.")
                else:
                    st.error("Username already exists!")

        else:  # Login
            if st.button("Login"):
                if username.strip() == "":
                    st.error("Username cannot be empty!")
                elif authenticate(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.login_time = time.time()  # Record login time
                    st.success(f"Logged in successfully as {username}!")
                    st.rerun()  # Refresh the app state
                else:
                    st.error("Invalid username or password")
    else:
        st.subheader(f"Welcome, {st.session_state.username}!")

        # Mental Health Prediction Section
        gender = st.selectbox("Gender", ["Male", "Female"])
        country = st.selectbox("Country", [
            'United States', 'Poland', 'Australia', 'Canada', 'United Kingdom',
            'South Africa', 'Sweden', 'New Zealand', 'Netherlands', 'India', 
            'Belgium', 'Ireland', 'France', 'Portugal', 'Brazil', 'Costa Rica', 
            'Russia', 'Germany', 'Switzerland', 'Finland', 'Israel', 'Italy', 
            'Bosnia and Herzegovina', 'Singapore', 'Nigeria', 'Croatia', 
            'Thailand', 'Denmark', 'Mexico', 'Greece', 'Moldova', 'Colombia', 
            'Georgia', 'Czech Republic', 'Philippines'
        ])
        occupation = st.selectbox("Occupation", ["Corporate", "Student", "Business", "Housewife", "Others"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        family_history = st.selectbox("Family History", ["Yes", "No"])
        treatment = st.selectbox("Treatment", ["Yes", "No"])
        days_indoors = st.selectbox("Days Indoors", ['1-14 days', 'Go out Every day', 'More than 2 months', '15-30 days', '31-60 days'])
        growing_stress = st.selectbox("Growing Stress", ["Yes", "No", "Maybe"])
        changes_habits = st.selectbox("Changes in Habits", ["Yes", "No", "Maybe"])
        mental_health_history = st.selectbox("Mental Health History", ["Yes", "No", "Maybe"])
        mood_swings = st.selectbox("Mood Swings", ["Low", "Medium", "High"])
        coping_struggles = st.selectbox("Coping Struggles", ["Yes", "No"])
        work_interest = st.selectbox("Work Interest", ["Yes", "Maybe", "No"])
        social_weakness = st.selectbox("Social Weakness", ["Yes", "No", "Maybe"])
        mental_health_interview = st.selectbox("Mental Health Interview", ["Yes", "Maybe", "No"])
        care_options = st.selectbox("Care Options", ["Yes", "No", "Not sure"])

        input_data = {
            "Gender": gender,
            "Country": country,
            "Occupation": occupation,
            "self_employed": self_employed,
            "family_history": family_history,
            "treatment": treatment,
            "Days_Indoors": days_indoors,
            "Growing_Stress": growing_stress,
            "Changes_Habits": changes_habits,
            "Mental_Health_History": mental_health_history,
            "Mood_Swings": mood_swings,
            "Coping_Struggles": coping_struggles,
            "Work_Interest": work_interest,
            "Social_Weakness": social_weakness,
            "mental_health_interview": mental_health_interview,
            "care_options": care_options
        }

        if st.button("Predict"):
            prediction, confidence_percentage = predict(input_data)  # Get both prediction and confidence
            status = map_to_status(prediction)
            st.write(f"Predicted Instability Rate (0 to 8): {prediction}")
            st.write(f"Confidence: {confidence_percentage:.2f}%")  # Display confidence percentage
            st.write(f"Mental Health Status: {status}")
            # Calculate time spent in seconds
            time_spent = int(time.time() - st.session_state.login_time)  # Duration in seconds

            # Save prediction result to the database
            save_prediction(st.session_state.username, datetime.datetime.now(), prediction, status, time_spent)

            # Fetch predictions for the user
            st.session_state.predictions = fetch_predictions(st.session_state.username)

        if st.button("Logout"):
            # Calculate time spent before logging out
            time_spent = int(time.time() - st.session_state.login_time)  # Duration in seconds
            save_prediction(st.session_state.username, datetime.datetime.now(), 0, "Logged Out", time_spent)  # Save logout record
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.login_time = None  # Reset login time
            st.session_state.predictions = pd.DataFrame(columns=["Date", "Prediction", "Status", "Time Spent"])  # Reset predictions on logout
            st.rerun()  # Refresh the app state

elif page == "Mood Tracking":
    st.subheader("Mood Tracking Records")

    # Fetch predictions for the user
    if st.session_state.logged_in:
        st.session_state.predictions = fetch_predictions(st.session_state.username)

        # Initialize visibility states
        if "show_records" not in st.session_state:
            st.session_state.show_records = False
        if "show_graph" not in st.session_state:
            st.session_state.show_graph = False

        # Buttons to toggle visibility
        if st.button("Show Mood Tracking Records"):
            st.session_state.show_records = not st.session_state.show_records

        if st.button("Show Mood Tracking Graph"):
            st.session_state.show_graph = not st.session_state.show_graph

        # Display mood tracking records
        if st.session_state.show_records:
            if not st.session_state.predictions.empty:
                st.write(st.session_state.predictions)
            else:
                st.write("No predictions recorded yet.")

        # Display mood tracking graph
        if st.session_state.show_graph:
            if not st.session_state.predictions.empty:
                # Count occurrences of each status
                status_counts = st.session_state.predictions['Status'].value_counts()

                # Plotting the graph
                plt.figure(figsize=(10, 5))
                plt.bar(status_counts.index, status_counts.values, color='blue', alpha=0.7)
                plt.title('Mental Health Status Distribution')
                plt.xlabel('Mental Health Status')
                plt.ylabel('Count')
                st.pyplot(plt)
            else:
                st.write("No predictions recorded yet.")
    else:
        st.warning("Please log in to access your mood tracking records.")

elif page == "Connect Page":
    st.subheader("Chat Room")

    # Check if the user is logged in
    if st.session_state.logged_in:
        # Security code input
        security_code = st.text_input("Enter Security Code", type="password")
        if st.button("Join Chat"):
            if security_code == "123456":  # Replace with your actual security code
                st.session_state.chat_active = True
                st.success("You have joined the chat room!")
            else:
                st.error("Invalid security code. Please try again.")

        if "chat_active" in st.session_state and st.session_state.chat_active:
            # Fetch and display previous chat messages
            messages = fetch_chat_messages()
            
            # Initialize last sender
            last_sender = None

            for message_id, username, message, timestamp in messages:
                if username == last_sender:
                    # Same user, display message without a new block
                    st.markdown(f"<div style='text-align: left;'><span style='color: gray;'>{timestamp}</span> - {message}</div>", unsafe_allow_html=True)
                else:
                    # Different user, display message on the right
                    st.markdown(f"<div style='text-align: right;'><strong>{username}</strong>: {message} <span style='color: gray;'>{timestamp}</span></div>", unsafe_allow_html=True)
                    last_sender = username  # Update last sender

                # Add delete button for the user's own messages
                if username == st.session_state.username:
                    if st.button("Delete", key=message_id):  # Use message_id as the key
                        delete_chat_message(message_id)  # Delete the message
                        st.success("Message deleted successfully.")
                        st.rerun()  # Refresh to show updated messages

            # Input for new messages
            new_message = st.text_input("Type your message here...")
            if st.button("Send"):
                if new_message.strip() != "":
                    save_chat_message(st.session_state.username, new_message)
                    st.rerun()  # Refresh to show new message
                else:
                    st.error("Message cannot be empty.")

            if st.button("Leave Chat"):
                st.session_state.chat_active = False
                st.success("You have left the chat room.")
    else:
        st.warning("Please log in to access the chat room.")


elif page == "Personalized Recommendations":
    st.subheader("Personalized Recommendations")

    if st.session_state.logged_in:
        # Fetch predictions for the user
        user_predictions = fetch_predictions(st.session_state.username)

        if not user_predictions.empty:
            # Analyze the user's mood status
            latest_status = user_predictions['Status'].iloc[-1]  # Get the latest status
            st.write(f"Your latest mental health status: {latest_status}")

            # Generate recommendations based on the latest status
            if latest_status == "Stable or Low Instability":
                recommendations = [
                    "Continue your daily routine and maintain healthy habits.",
                    "Engage in physical activities like walking or yoga.",
                    "Practice mindfulness or meditation for relaxation."
                ]
            elif latest_status == "Moderate Instability":
                recommendations = [
                    "Consider talking to a friend or family member about your feelings.",
                    "Try journaling to express your thoughts and emotions.",
                    "Engage in creative activities like drawing or music."
                ]
            elif latest_status == "High Instability or Severe Instability":
                recommendations = [
                    "Reach out to a mental health professional for support.",
                    "Practice deep breathing exercises to manage anxiety.",
                    "Limit exposure to stressful situations and take breaks."
                ]
            else:
                recommendations = ["No specific recommendations available."]
                
            # Display recommendations
            st.write("### Recommendations:")
            for rec in recommendations:
                st.write(f"- {rec}")
            st.subheader("Mental Health Resources and Support")

    # Provide links to various resources
            st.write("### Hotlines")
            st.write("[National Suicide Prevention Lifeline](https://suicidepreventionlifeline.org/) - Call 1-800-273-TALK (1-800-273-8255)")
            st.write("[Crisis Text Line](https://www.crisistextline.org/) - Text HOME to 741741")
            st.write("[SAMHSA National Helpline](https://www.samhsa.gov/find-help/national-helpline) - Call 1-800-662-HELP (1-800-662-4357)")

            st.write("### Online Therapy Options")
            st.write("[BetterHelp](https://www.betterhelp.com/) - Online therapy with licensed professionals.")
            st.write("[Talkspace](https://www.talkspace.com/) - Therapy via text, audio, and video messaging.")

            st.write("### Articles and Educational Materials")  
            st.write("[Mental Health America](https://www.mhanational.org/) - Resources and information on mental health.")
            st.write("[NAMI (National Alliance on Mental Illness)](https://www.nami.org/) - Information and support for mental health conditions.")


            st.write("### Local Mental Health Services")
            st.write("Find local mental health services in your area by visiting [Psychology Today](https://www.psychologytoday.com/us/therapists).")

        else:
            st.write("No mood tracking data available. Please make predictions first.")
            
    else:
        st.warning("Please log in to access personalized recommendations.")
        

elif page == "Admin Dashboard":
    # Admin login section
    if not st.session_state.admin_logged_in:
        admin_username = st.sidebar.text_input("Admin Username")
        admin_password = st.sidebar.text_input("Admin Password", type="password")
        if st.sidebar.button("Admin Login"):
            if authenticate(admin_username, admin_password):
                st.session_state.admin_logged_in = True
                st.success("Admin logged in successfully!")
            else:
                st.error("Invalid admin username or password")
    else:
        st.title("Admin Dashboard")

        # Password change section
        new_password = st.text_input("New Admin Password", type="password")
        confirm_password = st.text_input("Confirm New Admin Password", type="password")
        
        if st.button("Change Password"):
            if new_password == confirm_password:
                update_admin_password(new_password)
                st.success("Admin password updated successfully!")
            else:
                st.error("Passwords do not match!")

        # Fetch all users
        c.execute("SELECT username FROM users")
        users = c.fetchall()
        user_list = [user[0] for user in users]  # Extract usernames from tuples

        # Display total number of users in a large box
        st.markdown("<h2 style='text-align: center;'>Total Users</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{len(user_list)}</h1>", unsafe_allow_html=True)

        # User selection for mood history
        selected_user = st.selectbox("Select a user to view details", user_list)

        if selected_user:
            # Fetch predictions for the selected user
            user_predictions = fetch_predictions(selected_user)

            # Display user details in a table format
            if not user_predictions.empty:
                st.subheader(f"Details for {selected_user}")

                # Calculate user details
                total_predictions = len(user_predictions)
                average_status = user_predictions['Status'].value_counts().idxmax()  # Most common status
                total_time_spent = user_predictions['Time Spent'].sum() / 60  # Convert seconds to minutes
                last_prediction_date = user_predictions['Date'].max() if not user_predictions.empty else "N/A"
                last_prediction_weekday = pd.to_datetime(last_prediction_date).day_name() if last_prediction_date != "N/A" else "N/A"

                # Create a DataFrame for user details
                user_details_data = {
                    "Metric": [
                        "Total Predictions",
                        "Average Status",
                        "Last Prediction Date",
                        "Last Prediction Weekday",
                        "Total Time Spent (minutes)"
                    ],
                    "Value": [
                        total_predictions,
                        average_status,
                        last_prediction_date,
                        last_prediction_weekday,
                        total_time_spent  # Total time spent in minutes
                    ]
                }

                user_details_df = pd.DataFrame(user_details_data)

                # Display the user details in a table
                st.table(user_details_df)

                # Display mood tracking graph for the selected user
                status_counts = user_predictions['Status'].value_counts()
                plt.figure(figsize=(10, 5))
                plt.bar(status_counts.index, status_counts.values, color='blue', alpha=0.7)
                plt.title(f'Mental Health Status Distribution for {selected_user}')
                plt.xlabel('Mental Health Status')
                plt.ylabel('Count')
                st.pyplot(plt)
            else:
                st.write(f"No mood history recorded for {selected_user}.")


        # User Deletion Section
        st.subheader("Delete User")
        delete_user = st.selectbox("Select a user to delete", user_list)
        if st.button("Delete User"):
            if delete_user:
                c.execute("DELETE FROM users WHERE username=?", (delete_user,))
                conn.commit()
                st.success(f"User  '{delete_user}' has been deleted successfully.")
                # Refresh the user list after deletion
                c.execute("SELECT username FROM users")
                users = c.fetchall()
                user_list = [user[0] for user in users]  # Update user list
            else:
                st.error("Please select a user to delete.")
                
        # Overall user activity monitoring
        st.subheader("Overall User Activity Monitoring")

        # Fetch all predictions for overall statistics
        c.execute("SELECT username, date, prediction, status, time_spent FROM predictions")
        predictions = c.fetchall()
        predictions_df = pd.DataFrame(predictions, columns=["Username", "Date", "Prediction", "Status", "Time Spent"])

        # Convert the 'Date' column to datetime
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'], errors='coerce')

        if not predictions_df.empty:
            # Calculate overall statistics
            total_users_active = predictions_df['Username'].nunique()
            average_time_spent = predictions_df['Time Spent'].sum() / (total_users_active * 60)  # Average time spent in minutes
            most_active_weekday = predictions_df['Date'].dt.day_name().value_counts().idxmax()
    # Filter out "Logged Out" status
            filtered_predictions_df = predictions_df[predictions_df['Status'] != "Logged Out"]
            most_predicted_status = filtered_predictions_df['Status'].value_counts().idxmax() if not filtered_predictions_df.empty else "N/A"

            # Create a DataFrame for overall activity monitoring details
            overall_activity_data = {
                "Metric": [
                    "Total Active Users",
                    "Average Time Spent (minutes)",
                    "Most Active Weekday",
                    "Most Predicted Status"
                ],
                "Value": [
                    total_users_active,
                    average_time_spent,  # Average time spent in minutes
                    most_active_weekday,
                    most_predicted_status
                ]
            }

            overall_activity_df = pd.DataFrame(overall_activity_data)

            # Display the overall activity monitoring details in a table
            st.table(overall_activity_df)

            active_users_by_weekday = predictions_df.groupby(predictions_df['Date'].dt.day_name())['Username'].nunique().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0)

    # Buttons for displaying graphs
            if st.button("Show Total Active Users by Weekday Graph"):
                plt.figure(figsize=(10, 5))
                plt.bar(active_users_by_weekday.index, active_users_by_weekday.values, color='blue', alpha=0.7)
                plt.title('Total Active Users by Weekday')
                plt.ylabel('Count of Active Users')
                plt.xlabel('Weekday')
                plt.xticks(rotation=45)
                st.pyplot(plt)


            if st.button("Show Most Active Weekday Graph"):
                weekday_counts = predictions_df['Date'].dt.day_name().value_counts()
                plt.figure(figsize=(10, 5))
                plt.plot(weekday_counts.index, weekday_counts.values, marker='o', color='blue', alpha=0.7)
                plt.title('Most Active Weekday')
                plt.xlabel('Weekday')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(plt)

            if st.button("Show Most Predicted Status Graph"):
                status_counts = predictions_df['Status'].value_counts()
                plt.figure(figsize=(10, 5))
                plt.plot(status_counts.index, status_counts.values, marker='o', color='blue', alpha=0.7)
                plt.title('Most Predicted Status')
                plt.xlabel('Status')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                st.pyplot(plt)

        else:
            st.write("No predictions recorded yet.")