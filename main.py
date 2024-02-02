import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('fake_news.csv')
list(dataset.columns)
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

x = dataset['text']
y = dataset['label']

# разделение на обучающий и тестовый наборы
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# преобразование текстовых данных в матрицу TF-IDF признаков
tfidf_vectorizer = TfidfVectorizer()
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xtest_tfidf = tfidf_vectorizer.transform(xtest)

# создание и обучение модели Passive Aggressive Classifier
model = PassiveAggressiveClassifier()
model.fit(xtrain_tfidf, ytrain)

# оценка производительности модели
print('Точность: ', model.score(xtest_tfidf, ytest))

# демонстрация работоспособности модели
news_headline = 'US threatens to reimpose sanctions on Venezuelan oil sector'
data = tfidf_vectorizer.transform([news_headline])
print(model.predict(data))

# получаем предсказания модели на тестовом наборе
pred = model.predict(xtest_tfidf)

# вычисляем матрицу ошибок
conf_matrix = confusion_matrix(ytest, pred)

# визуализируем матрицу ошибок с помощью seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
