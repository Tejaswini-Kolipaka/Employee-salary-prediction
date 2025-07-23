from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import joblib

# Use raw string or forward slashes in path
df = pd.read_csv(r"C:\Users\Sravanthichalla\Documents\adult 3.csv")

# Print columns to check
print("Columns in dataset:", df.columns.tolist())

# Make sure your mappings match your dataset's actual categories
education_map = {
    "HS-grad": 0, "Some-college": 1, "Bachelors": 2,
    "Masters": 3, "PhD": 4, "Assoc": 5
}

occupation_map = {
    "Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3,
    "Exec-managerial": 4, "Prof-specialty": 5, "Handlers-cleaners": 6,
    "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
    "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12,
    "Armed-Forces": 13
}

# Map categorical columns — make sure these columns exist in your CSV
df['education'] = df['education'].map(education_map)
df['occupation'] = df['occupation'].map(occupation_map)

# Adjust this list according to your dataset columns
features = ['age', 'education', 'occupation', 'hours-per-week']  # Removed 'experience' for now

X = df[features]
y = df['salary_class']  # Make sure 'salary_class' is the correct target column name

model = GradientBoostingClassifier()
model.fit(X, y)

joblib.dump(model, "best_model.pkl")
print("Model trained and saved successfully!")

