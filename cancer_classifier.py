from sklearn import datasets, model_selection
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

breast_cancer_data = datasets.load_breast_cancer()

training_data, validation_data, training_labels, validation_labels = model_selection.train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

accuracies: int = []
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(range(1, 101), accuracies)
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('Breast Cancer Classifier')
plt.show()
