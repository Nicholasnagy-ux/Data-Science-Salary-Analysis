import pandas as pd
import pyarrow as pa
from sklearn.model_selection import train_test_split
import sklearn.tree as tree

TEST_SIZE = 0.2
NUM_OF_TESTS = 100
class ClassificationModel:
    def __init__(self):
        self.model = tree.DecisionTreeClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)

def Model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    classificationModel = ClassificationModel()
    classificationModel.train(x_train, y_train)
    predictions = classificationModel.predict(x_test)
    accuracy_score = classificationModel.evaluate(x_test, y_test)
    return accuracy_score

salaryData = pd.read_csv('data/ds_salaries.csv')
pd.set_option('display.max_columns', None)
print(salaryData)
y = salaryData['employment_type']
x = salaryData[['salary_in_usd', 'remote_ratio']].astype(float)
accuracy_score = 0
for i in range(NUM_OF_TESTS):
    accuracy_score += Model(x, y)
accuracy_score /= NUM_OF_TESTS
print('The average accuracy for %s tests of the classification model is: %s' %(NUM_OF_TESTS, accuracy_score))