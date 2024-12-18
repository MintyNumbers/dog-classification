# Klassifikation von Hunderassen mit klassischen ML sowie mit DL Methoden



## Inhaltsverzeichnis

1. [Beschreibung](#beschreibung)
2. [Projektausführung](#projektausführung)
3. [Third Example](#third-example)
4. [Fourth Example](#fourth-examplehttpwwwfourthexamplecom)



## Beschreibung

Dieses Projekt implementiert zwei unterschiedliche Ansätze zur Klassifikation von Bildern von Hunden nach ihrer Rasse. Die zwei gewählten Ansätze sind dabei:
1. Klassisches Machine Learning: Merkmalsbasierte Klassifikation mit Support Vector Machines und k-Nearest Neighbor Klassifikator
2. Deep Learning: Convolutional Neural Network und Transfer Learning



## Projektausführung

### Dependencies
Das Projekt benötigt Python 3 mit folgenden Bibliotheken:
- keras
- matplotlib
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


## Authors

* [Oliver Gassmann](https://github.com/olivergassmann)
  * `helpers/split.py`
  * `smv.ipynb`
  * `helpers/various_features.py`

* [Maria Vladykin](https://github.com/MintyNumbers/)
  * `helpers/split.py`
  * `deeplearning.ipynb`
  * `helpers/cnn.py`
  * `helpers/dataloader.py`
  * `helpers/train.py`
