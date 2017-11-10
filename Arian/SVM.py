from sklearn.svm import SVR

class TwitterSVM():

    def __init__(self, input, output):

        self.input_data = input
        self.output_data = output
        self.classifier = SVR(C=1.0, epsilon=0.2)
        self.classifier.fit(input, output)

    #end init

    def predict(self, X):

        result = self.classifier.predict(X)
        return result

    #end predict






