import os
import pandas as pd 
# import statements for f1 score, precision score, recall score, confusion matrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(__file__),'..','.env')
load_dotenv(env_path)

def fraud_detection_train(file_path):
    df = pd.read_csv(file_path)

    df.dropna(inplace=True)

    x = df.drop('is_fraud',axis=1)
    y = df['is_fraud']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = (y_test == y_pred).mean()
    print(f"Model accuracy: {accuracy}")

    f1 = f1_score(y_test, y_pred)
    print(f"Model F1 score: {f1}")

    precision = precision_score(y_test, y_pred)
    print(f"Model precision: {precision:.2f}")

    recall = recall_score(y_test, y_pred)
    print(f"Model recall: {recall:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    file_path = os.getenv('file_path')
    fraud_detection_train(file_path)