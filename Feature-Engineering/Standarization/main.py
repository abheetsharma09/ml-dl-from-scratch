class Standarization:
    def __init__(self ,lst):
        self.lst = lst

    def calculate(self):
        #calculations for variance
        varianceResultsqSum = 0
        totalMean = 0 
        totalSum = 0
        standarizedLst = []
        for i in self.lst:
            totalSum = totalSum + i
        totalMean = totalSum / len(self.lst)
        for j in self.lst:
            varianceResultsqSum = varianceResultsqSum +((j-totalMean) ** 2)
        variance = varianceResultsqSum / len(self.lst)
        standardDeviation = variance ** 0.5
        for k in self.lst:
            standarizedLst.append((k - totalMean)/standardDeviation)

        return standarizedLst
    
test_data = [
    73.52, 14.89, 44.12, 92.05, 5.11, 67.84, 31.45, 88.91, 12.33, 53.67,
    99.14, 25.40, 61.22, 8.76, 47.93, 83.19, 36.54, 70.08, 19.65, 55.42,
    41.18, 96.73, 2.34, 64.51, 28.09, 85.66, 11.02, 50.88, 77.23, 22.91,
    58.70, 6.44, 49.33, 91.17, 33.82, 79.50, 15.26, 63.11, 87.04, 18.39,
    46.05, 94.21, 4.88, 68.12, 29.74, 82.59, 13.90, 52.41, 75.99, 21.06,
    40.23, 97.85, 1.15, 66.39, 26.47, 84.13, 9.88, 51.56, 78.42, 24.33,
    57.12, 7.89, 48.01, 90.62, 35.19, 81.34, 16.71, 62.45, 86.29, 17.04,
    43.88, 93.56, 3.22, 65.04, 30.61, 89.47, 10.55, 54.18, 76.81, 23.50,
    59.94, 5.72, 45.16, 95.12, 32.07, 80.22, 14.48, 60.77, 88.01, 19.92,
    42.61, 98.34, 0.45, 69.57, 27.83, 83.68, 11.74, 55.91, 74.25, 20.66
]

print(Standarization(test_data).calculate())


