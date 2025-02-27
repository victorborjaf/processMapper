import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings
from sklearn.metrics import classification_report, confusion_matrix
from gaussClassifier import *
from gaussRegression import *

def addSum(multivect):
    sumv = np.sum(multivect, axis=1)
    sumv = sumv.reshape((len(sumv), 1))
    return np.concatenate((multivect, sumv), axis=1)

class PreparationParameters():
    def __init__(self, inputDim=3, outputFracture=True, outputPermea=True, sumporogens=False):
        self.inputDim = inputDim
        self.outputFracture = outputFracture
        self.outputPermea = outputPermea
        self.sumporogens = sumporogens

class PCDataSet():
    def __init__(self, path=None):
        self.missing_value_mask = None
        self.multiVector = self.load_filtered_csv(path) if path else np.zeros((0, 11))
        self.nbVectors = np.shape(self.multiVector)[0]

    def load_filtered_csv(self, path):
        try:
            with open(path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)
                
                valid_data = []
                for row in reader:
                    if row[0].strip().startswith('#'):
                        continue
                    row = [float(val) if val.strip() else np.nan for val in row]
                    while len(row) < 11:
                        row.append(np.nan)  # Ensure each row has exactly 11 columns
                    valid_data.append(row)

            data = np.array(valid_data, dtype=float)
            self.missing_value_mask = np.isnan(data)
            return np.nan_to_num(data, nan=np.nanmean(data, axis=0))  # Fill NaNs with column mean
        except Exception as e:
            warnings.warn(f"Error loading CSV: {e}")
            return np.zeros((0, 11))

    def addFromCsv(self, path):
        newdata = np.genfromtxt(path, skip_header=1, delimiter=',', comments='#', filling_values=np.nan)
        if newdata.ndim == 1:
            newdata = np.reshape(newdata, (1, -1))

        newdata = np.pad(newdata, [(0, 0), (0, max(0, 11 - newdata.shape[1]))], constant_values=np.nan)[:, :11]
        self.missing_value_mask = np.isnan(newdata)
        newdata = np.nan_to_num(newdata, nan=np.nanmean(newdata, axis=0))
        
        self.multiVector = np.concatenate((self.multiVector, newdata), axis=0)
        self.nbVectors = np.shape(self.multiVector)[0]
    
    def switch_columns(self, col1, col2):
        if 0 <= col1 < self.multiVector.shape[1] and 0 <= col2 < self.multiVector.shape[1]:
            self.multiVector[:, [col1, col2]] = self.multiVector[:, [col2, col1]]
        else:
            warnings.warn("Invalid column indices provided for swapping.")
    
    def prepareClassification(self, params=None, selected_cols=None):
        if params is None:
            params = PreparationParameters()

        if selected_cols is None:
            selected_cols = range(params.inputDim)
        
        self.inputVectors = self.multiVector[:, selected_cols] / 100.0
        if params.sumporogens:
            self.inputVectors = addSum(self.inputVectors)
        
        self.classVector = self.multiVector[:, 8].astype(bool)  # Permeability class
        self.permeabilityValues = self.multiVector[:, 10]
    
    def plot(self, figName=None, show=True):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.inputVectors[:, 0], self.inputVectors[:, 1], c=self.classVector, cmap='coolwarm', alpha=0.7)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Classification Results")
        if figName:
            plt.savefig(figName)
        if show:
            plt.show()
    
    def evaluate_classification(self, true_labels):
        print("\nClassification Report:")
        print(classification_report(true_labels, self.classOutput.round()))
        print("Confusion Matrix:")
        print(confusion_matrix(true_labels, self.classOutput.round()))
    
    def initializeClassification(self, paramsC=None, paramsR=None, classifier=None):
        self.classifier = classifier if classifier else gaussClassifier(paramsC, paramsR)
        if not hasattr(self.classifier.regression.params, "corrlenVect"):
            self.classifier.regression.isotropic_corrlen(self)
    
    def classify(self, queryPts, paramsC=None, paramsR=None, classifier=None):
        self.queryPts = queryPts
        self.nbQuery = np.shape(self.queryPts)[0]
        if self.prepaParameters.sumporogens:
            self.queryPts = addSum(self.queryPts)
        self.classifier.classify(self)
        self.classOutput = 1 / (1 + np.exp(-np.clip(self.classifier.regression.regsOutput, -100, 100)))
    
    def optimizeLogMLClassif(self, paramsC=None, paramsR=None, classifier=None):
        self.classifier = classifier if classifier else gaussClassifier(paramsC, paramsR)
        self.classifier.optimizeLogML(self)
    
if __name__ == "__main__":
    data = PCDataSet('samples/initial_samples.csv')
    data.addFromCsv('samples/samples.csv')
    params = PreparationParameters()
    data.prepareClassification(params)
    data.plot()
    data.evaluate_classification(data.classVector)
