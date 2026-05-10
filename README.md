# 🧠 Smart Career Recommendation System

## 📌 Overview
This project is an AI-powered **Career Recommendation System** that suggests suitable career paths based on a user's:
- Skills  
- Interests 
- Tech Tools

- Educational background  

It uses machine learning techniques (TF-IDF and similarity matching) to recommend careers that best match user input.

---

## 🚀 Features
- 🔍 Personalized career recommendations  
- 📊 TF-IDF vectorization and cosine similarity  
- 🧠 Multi-factor input (skills, interests, education)  
- 🌐 Interactive web app built with Streamlit  

---

## 🛠️ Technologies Used
- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  

---

## 📂 Project Structure
```bash
smart_career_models/
│
├── smart_career_app.py              # Streamlit app
├── Smart career recommendation.ipynb  # Model development
├── *.pkl                           # Trained models
├── AI-based Career Recommendation System.csv
├── requirements.txt
└── README.md
```
⚙️ How It Works
User inputs:
Skills
Interests
Education
Input is transformed using TF-IDF
Cosine similarity is computed
System recommends the closest matching careers
▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/Jaypee88/Smart-Tech-career-recommender.git
cd Smart-Tech-career-recommender
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run smart_career_app.py
📊 Dataset

The dataset contains structured career-related data such as:

Required skills
Educational level
Career Interests
📈 Future Improvements
Add real-time job market data
Improve model accuracy
Add user authentication
Deploy as a full web application
👤 Author

Nwankwo Chidozie Johnpaul

📜 License

This project is for educational purposes.

To access the app follow the link below
https://smart-tech-career-recommender-nbrwnq2usx6a5ojhetvmoo.streamlit.app/

