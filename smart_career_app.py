#Import the necessary libraries
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data
#Defined a function that implies loading the dataset also giving a threshold  for the dataset
def load_data(path):
    df = pd.read_csv(path)
    # ensure expected columns exist
    expected = ['Skills', 'Interests', 'Education', 'Recommended_Career','Tools']
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    return df


# Load your saved vectorizers and matrices
tfidf_skills = joblib.load("tfidf_skills.pkl")
tfidf_interests = joblib.load("tfidf_interests.pkl")
tfidf_edu = joblib.load("tfidf_edu.pkl")
tfidf_tools = joblib.load("tfidf_tools.pkl")


skills_matrix = joblib.load("skills_matrix.pkl")
interests_matrix = joblib.load("interests_matrix.pkl")
edu_matrix = joblib.load("edu_matrix.pkl")
tools_matrix = joblib.load("tools_matrix.pkl")


#Defined the  columns for which the recommended dataframe  should have
recommended_df = pd.DataFrame(columns=['Career', 'Skills Match %', 'Interests Match %',
                            'Education Match %','Tools Match %', 'Final Score %'])


#Defined a function that will  do the recommendation
def recommend(df, tfidf_skills, tfidf_interests, tfidf_edu, tfidf_tools,
                  skills_matrix, interests_matrix, edu_matrix, tools_matrix,
                  user_skills, user_interests, user_edu, user_tools,
                  weights=(0.5, 0.2, 0.3), top_n=5):

    #Used the loaded tfid vec to vectorise some of the user inputs
    u_skills_vec = tfidf_skills.transform([user_skills])
    u_interests_vec = tfidf_interests.transform([user_interests])
    u_edu_vec = tfidf_edu.transform([user_edu])
    u_tools_vec = tfidf_tools.transform([user_tools])

    # Compute cosine similarity
    skills_sim = cosine_similarity(u_skills_vec, skills_matrix).flatten()
    interests_sim = cosine_similarity(u_interests_vec, interests_matrix).flatten()
    edu_sim = cosine_similarity(u_edu_vec, edu_matrix).flatten()
    tools_sim = cosine_similarity(u_tools_vec, tools_matrix).flatten()

    # Weighted final score
    final_score = ((weights[0] * skills_sim) + (weights[1] * interests_sim)
                   + (weights[2] * edu_sim)+ (weights[3] * tools_sim))

    # Assigned how the  dataframe for the recommended df should be and  contain
    recommended_df = pd.DataFrame({
        'Career': df['Recommended_Career'],
        'Skills Match %': (skills_sim*100).round(2),
        'Interests Match %': (interests_sim*100).round(2),
        'Education Match %': (edu_sim*100).round(2),
        'Tools Match %': (tools_sim*100).round(2),
        'Final Score %': (final_score*100).round(2)
    })
#Dropped duplicates for the recommended dataframe and sorted by final scoring returning top idx
    recommended_df = recommended_df.drop_duplicates(subset=['Career'])
    recommended_df = recommended_df.sort_values(by='Final Score %', ascending=False)
    recommended_df = recommended_df.reset_index(drop=True)
    top_recs = recommended_df.head(top_n)
    return top_recs

#Give the recommended career
st.subheader("Top Recommended Careers")
st.dataframe(recommended_df)

# A function to Plot a graph to show the top recommended careers with their similarity scores
def plot_bar(recommended_df):
    st.subheader("Visualizing Recommendations")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x='Career', y='Final Score %', data=recommended_df, ax=ax)
    plt.xticks(rotation=75)
    st.pyplot(fig)

#Features to improve the UI/UX of the app
st.title("Smart Career Recommendation System")
st.markdown("Use the sidebar to navigate. This app recommends tech careers based on"
            " skills, interests, tech tools and education.")
page = st.sidebar.selectbox('Choose page', ['Home', 'Explore Dataset', 'Recommend'])


#Created a sidebar for dataset where one can go to load the dataset before prediction
with st.sidebar.expander('Dataset'):
        Dataset = st.sidebar.checkbox('Load Dataset')
        if Dataset:
            df = load_data("tech_skills_cleaned.csv")
        else:
            st.info('Please "Load sample CSV"')
            st.stop()


st.sidebar.markdown('Weight scoring for recommendation')
#Gave the user the power to coose the feature or weight of them for the  prediction
weights_skill = st.sidebar.slider('Skills weight', 0.0, 1.0, 0.4)
weights_interest = st.sidebar.slider('Interests weight', 0.0, 1.0, 0.3)
weights_edu = st.sidebar.slider('Education weight', 0.0, 1.0, 0.1)
weights_tools = st.sidebar.slider('Tools weight', 0.0, 1.0, 0.2)
w_sum = weights_skill + weights_interest + weights_edu +weights_tools
if w_sum == 0:
    weights = (0.4, 0.3, 0.1, 0.2) #A constant value  incase  user  decides not to choose the weight

else:
    #This is to normalise the weight so that no matter the user input they should all sum  to 1
    weights = (weights_skill/w_sum, weights_interest/w_sum, weights_edu/w_sum, weights_tools/w_sum)
    st.write(f'''Normalized Weights: Skills={weights[0]:.2f}, Interests={weights[1]:.2f},
             Education={weights[2]:.2f}, Tools={weights[3]:.2f}''')


if  page == 'Home':
    st.header("Homepage")
    st.write('''Welcome this app recommends a career path to users based on their interests,
             skills, tech tools and education''')

#Added a page that allows  a user to know about the dataset and explore for some of its features
elif page == 'Explore Dataset':
    st.header('DATASET OVERVIEW')

    st.write(f'ROWS: {len(df)}')#lenght of the dataset

    st.write(f'COLUMNS: {len(df.columns)}') #display the columns

    st.write(f'information:{df.info}') #infomation about it

    st.write(f'data head:{df.head()}') # display the head of the dataset

#Defined the  page for recommendation
elif page == 'Recommend':
    st.header(' GET CAREER RECOMMENDATIONS')
    #Defined  and now accepted some  user inputs
    user_skills = st.sidebar.text_input("Enter your Skills").strip().lower()
    user_interests = st.sidebar.text_input("Enter your Interests").strip().lower()
    user_tools =st.sidebar.text_input("Enter tech tools you are conversant with").strip().lower()
    Education_options = ['Postgraduate', 'Graduate', 'Undergraduate']
    user_education = st.selectbox('Education', Education_options)
    top_n = st.number_input('How many recommendations to show?',min_value=1,max_value=10,value=5)

    if st.button('Recommend Career'):
        if not user_skills or not user_interests or not user_education or not user_tools:
            st.error('Please fill out the required fields')

        else:
            recs = recommend(df, tfidf_skills, tfidf_interests, tfidf_edu, tfidf_tools,
                             skills_matrix, interests_matrix, edu_matrix, tools_matrix,
                             user_skills, user_interests, user_education, user_tools,
                             weights=weights, top_n=top_n)

#Called the function to recommend and all other functions predefined before
            if recs is not None and not recs.empty:
                st.subheader("Top Recommended Careers")
                st.dataframe(recs)
                st.subheader('Visual representations')
                plot_bar(recs)
                csv = recs.to_csv(index=False).encode('utf-8')
                st.download_button("Download Recommendations", data=csv,
                                   file_name="career_recommendations.csv")
            else:
                st.warning("No matching careers found.")




