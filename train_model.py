from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Binary: Setosa or not

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
