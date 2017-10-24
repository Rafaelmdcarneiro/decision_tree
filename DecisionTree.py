import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

def plot_cm(cm, cm_norm):
    plt.figure()
    plt.title(u'Matriz de Confusão')

    a = plt.subplot(121)
    a.set_title(u"Matriz de Confusão Regular", fontsize=18)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.ylabel(u'Classe Verdadeira', fontsize=16)
    plt.xlabel(u'Classe Estimada', fontsize=16)

    b = plt.subplot(122)
    b.set_title(u"Matriz de Confusão Normalizada", fontsize=18)
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.ylabel(u'Classe Verdadeira', fontsize=16)
    plt.xlabel(u'Classe Estimada', fontsize=16)

    plt.tight_layout()
    plt.show()

# Importa o banco de dados Iris
iris = datasets.load_iris()
X = iris.data
Y = iris.target
model = DecisionTreeClassifier()

model.fit(X, Y)

Y_pred = model.predict(X)

score = model.score(X, Y)
print(u"Score: {0:.2f}").format(score)

cm = confusion_matrix(Y, Y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(precision=2)
print(u'Matriz de Confusão Regular')
print(cm)
print(u'Matriz de Confusão Normalizada')
print(cm_norm)

plot_cm(cm, cm_norm)