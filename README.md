# Klassifikation von Hunderassen mit klassischen ML Methoden und mit DL Methoden


## Inhaltsverzeichnis

- [Klassifikation von Hunderassen mit klassischen ML Methoden und mit DL Methoden](#klassifikation-von-hunderassen-mit-klassischen-ml-methoden-und-mit-dl-methoden)
  - [Inhaltsverzeichnis](#inhaltsverzeichnis)
  - [Beschreibung](#beschreibung)
  - [Projektausführung](#projektausführung)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Ausführung](#ausführung)
  - [Vergleich der Methoden](#vergleich-der-methoden)
  - [Authors](#authors)


## Beschreibung

Dieses Projekt implementiert zwei unterschiedliche Ansätze zur Klassifikation von Bildern von Hunden nach ihrer Rasse. Die zwei gewählten Ansätze sind dabei:
1. Klassisches Machine Learning: Merkmalsbasierte Klassifikation mit Support Vector Machines und k-Nearest Neighbor Klassifikator
2. Deep Learning: Convolutional Neural Network und Transfer Learning


## Projektausführung

### Dependencies
Das Projekt benötigt Python 3 mit folgenden Bibliotheken:
- matplotlib
- notebook
- numpy
- scikit-image
- scikit-learn
- seaborn
- tensorflow


### Installation

In einem beliebigen Ordner soll man das Repository mit folgendem Befehl klonen:
```bash
git clone git@github.com:MintyNumbers/dog-classification.git
```
Anschließend soll man noch den Datensatz http://vision.stanford.edu/aditya86/ImageNetDogs/ in den `./dataset` Ordner (der vorher erstellt werden msuss) herunterladen. Anschließend soll die erste Codezelle des `svm.ipynb` Notebooks ausgeführt werden, um fünf Hunderassen zu selektieren, einen Train/Test Split durchzuführen und die Daten dann in den `./dataset/Train` und den `./dataset/Test` Unterordner zu kopieren.


### Ausführung

* Zunächst sollte das `svm.ipynb` Notebook ausgeführt werden
* Dann sollte das `deeplearning.ipynb` Notebook ausgeführt werden


## Vergleich der Methoden

Beim klassischen Machine Learning konnte keine der beiden untersuchten Methoden (SVM und k-NN) eine gute Accuracy erzielen (immer < 50 %). Das liegt daran, dass das klassische Machine Learning vor allem bei einfach trennbaren Klassen sehr gute Ergebnisse erzielen kann. Hunderassen sind hingegen nicht trivial trennbar, sondern haben viele und große Überlappungen, Darüber hinaus ist es sehr zeitaufwendig, aussagekräftige Merkmale für die Bilder zu finden, die die Genauigkeit weiter erhöhen könnten. Ein Beispiel dafür ist der Gabor-Filter, der ursprünglich dafür vorgesehen war, Fellstrukturen zu erkennen. Aufgrund einer extrem langen Rechenzeit der ausprobierten Funktion konnte dieses Merkmal jedoch nicht sinnvoll eingesetzt werden. Eine weitere Möglichkeit zur zukünftigen Verbesserung der Ergebnisse wäre eine Erkennung der Position des Hundes, sodass die Features nicht durch den Hintergrund verzerrt werden.\
An dieser Stelle bietet das Deep Learning einen deutlichen Vorteil, weil es nicht auf bereits extrahierte Merkmale angewiesen ist.

Im Gegensatz zu einem klassischen ML Modell, das auf Merkmalsextraktion beruht, ist ein DL Modell i.d.R. eine Black-Box. Das bedeutet, dass durch komplexe mathematische Berechnungen, Zusammenhänge in den Daten gefunden werden, mit dessen Hilfe man die Trainingsdaten möglichst präzise klassifizieren kann. So werden auch Merkmale der Bilder berücksichtigt, auf die man vielleicht im ersten Moment beim Entwurf eines klassischen Modells nicht gekommen wäre.\
Dennoch wurde im Rahmen der Bearbeitung auch klar, dass trotz Regularisierung und Datenaugmentierung das Modell stark zu Overfitting geneigt war. Aufgrund der Black-Box Natur des Modells ist es aber schwer die Gründe dafür herauszufinden - aber mithilfe von Explainability Methoden wie LIME, SHAP oder GradCAM könnte man das Verhalten möglicherweise besser analysieren. Des weiteren wäre auch eine Optimierung von Hyperparametern (wie die Learning Rate, die Batch Size, den gewählte Optimizer oder die Höhe und Art der Regularisierung) oder eine umfangreichere Datenlage und/oder Datenaugmentierung ein möglicher Weg, die Performance des Modells zu verbessern.


## Authors

* [Oliver Gassmann](https://github.com/olivergassmann)
  * `helpers/split.py`
  * `svm.ipynb`
  * `helpers/various_features.py`

* [Maria Vladykin](https://github.com/MintyNumbers/)
  * `helpers/split.py`
  * `deeplearning.ipynb`
  * `helpers/cnn.py`
  * `helpers/dataloader.py`
  * `helpers/train.py`
