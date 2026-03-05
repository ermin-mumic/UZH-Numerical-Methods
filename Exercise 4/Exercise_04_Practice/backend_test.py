import numpy as np
import sys
import backend
import traceback

class Tester:
    def __init__(self):
        self.module = None

    def create_test_matrix(self, sign=1):
        A = np.triu(np.ones((10, 10)))
        for i in range(10):
            A[i] *= (i + 1) / 10.0
            A[:, i] *= (10 - i + 1) / 10.0
        A = A.transpose().dot(A).dot(A) * sign * np.pi / np.e
        return A

    def create_b(self):
        return np.ones(10)

    #############################################
    # Task a
    #############################################

    def testA(self, l: list, task):
        comments = ""

        def evaluate(A, b):
            nonlocal comments
            try:
                eigenVector = self.module.powerMethod(np.copy(A), np.copy(b))
                eigenVector /= np.linalg.norm(eigenVector)
                vals, reference = np.linalg.eig(A)
                maxReference = reference[:, np.argmax(np.abs(vals))]
                dot = eigenVector.dot(maxReference)
                if 1 - np.abs(dot) < 1e-6:
                    comments += "passed. "
                else:
                    comments += "failed. Dot product exceeded tolerance value. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Positive case
        comments += "Positive case "

        def create_test_matrix(sign=1):
            A = np.triu(np.ones((10, 10)))
            for i in range(10):
                A[i] *= (i + 1) / 10.0
                A[:, i] *= (10 - i + 1) / 10.0
            A = A.transpose().dot(A).dot(A) * sign * np.pi / np.e
            return A

        A = create_test_matrix()
        b = self.create_b()
        evaluate(A, b)

        # Negative case
        comments += "Negative case "

        A = create_test_matrix(-1)
        evaluate(A, b)

        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task b
    #############################################

    def testB(self, l: list, task):
        comments = ""

        def evaluate(A, b):
            nonlocal comments
            try:
                eigenVector = self.module.inversePowerMethod(np.copy(A), np.copy(b))
                eigenVector /= np.linalg.norm(eigenVector)
                vals, reference = np.linalg.eig(A)
                minReference = reference[:, np.argmin(np.abs(vals))]
                dot = eigenVector.dot(minReference)
                dot /= np.linalg.norm(eigenVector)
                if 1 - np.abs(dot) < 1e-6:
                    comments += "passed. "
                else:
                    comments += "failed. Dot product exceeded tolerance value. "
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # Positive case
        comments += "Positive case "

        A = self.create_test_matrix()
        b = self.create_b()
        evaluate(A, b)

        # Negative case
        comments += "Negative case "

        A = self.create_test_matrix(-1)
        evaluate(A, b)

        result = [task, comments]
        print(result)
        l.extend(result)

    #############################################
    # Task c
    #############################################

    def testC(self, l: list, task):
        comments = ""

        def evaluate(X, Y):
            nonlocal comments
            try:
                eigenVector = self.module.linearPCA(np.copy(X), np.copy(Y))
                eigenVector /= np.linalg.norm(eigenVector)
                data = np.array([X, Y]).T
                data -= np.mean(data, axis=0)
                vals, reference = np.linalg.eig(data.T.dot(data))
                referenceVector = reference[:, np.argmax(np.abs(vals))]
                dot = eigenVector.dot(referenceVector)
                dot /= np.linalg.norm(eigenVector)
                if 1 - np.abs(dot) < 1e-6:
                    comments += "passed. "
                else:
                    comments += "failed. Dot product exceeded tolerance value."
            except Exception as e:
                comments += "crashed. " + str(e) + " "
                tb = traceback.extract_tb(sys.exc_info()[2])[-1]
                fname = str(tb.filename.split("/")[-1])
                lineno = str(tb.lineno)
                comments += "Here: " + str(fname) + ":" + str(lineno) + " "

        # X case
        comments += "X case "

        X = np.linspace(5, 15, 100)
        Y = (np.random.random(100) - 0.5) * 3.0 + 100
        evaluate(X, Y)

        # Y case
        comments += "Y case "

        X = (np.random.random(100) - 0.5) * 3.0 + 100.0
        Y = np.linspace(5, 15, 100)
        evaluate(X, Y)

        # XY case
        comments += "XY case "

        X = np.linspace(5, 15, 100)
        Y = np.copy(X) + (np.random.random(100) - 0.5) * 3.0 + 100
        evaluate(X, Y)

        result = [task, comments]
        print(result)
        l.extend(result)

    def performTest(self, func, task):
        l = []
        try:
            func(l, task)
            return l
        except Exception as e:
            return [task, "Interrupted due to exception: " + str(e)]

    def runTests(self, module, l):
        self.module = module

        def evaluateResult(task, result):
            if len(result) == 0:
                l.append([task, 0, "Interrupted."])
            else:
                l.append(result)

        result = self.performTest(self.testA, "4.1a)")
        evaluateResult("4.1a)", result)

        result = self.performTest(self.testB, "4.1b)")
        evaluateResult("4.1b)", result)

        result = self.performTest(self.testC, "4.1c)")
        evaluateResult("4.1c)", result)

        return l


tester = Tester()
overall_result = []
tester.runTests(backend, overall_result)
