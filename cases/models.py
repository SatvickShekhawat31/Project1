import joblib

# Load models
logistic_model = joblib.load('models/court_case_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Print logistic model coefficients
print("Logistic Model Coefficients:")
print(logistic_model.coef_)
print("Model Intercept:")
print(logistic_model.intercept_)
print("Classes in the model:")
print(logistic_model.classes_)

# Print TF-IDF vectorizer vocabulary
print("TF-IDF Vectorizer Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())
