import csv

train = []
test = []
classValues = []
classValuesCount = {}
attributeValues = []
attributeValueCount = []
probabilitiesList = []
attributeClassProbabilities = []
classProbabilities = []
highestValue = []

def clearAll():
    classValues.clear()
    attributeValues.clear()
    probabilitiesList.clear()
    classProbabilities.clear()
    train.clear()
    test.clear()
    classValuesCount.clear()
    attributeClassProbabilities.clear()

"""Read data"""

dataset = 'Datasets/voting.csv' # Change dataset name to test a different dataset.

with open(dataset) as f:
    reader = csv.reader(f)
    data = list(reader)

"""Split data into train/test sets"""

count = 0
index = 0
total = len(data)-1
numOfTrain = round(total*0.9) # % of train data
numOfTest = round(total*0.1) # % of test data
attributeNames = data[0]
totalAcc = 0
testIndex = 1
data.remove(attributeNames) # Remove names so they don't get appended into train/tests

k = 1
while k < 11:   #   Iterate till 10 folds

    #   Adding to test list
    while count < numOfTest-1:
        selected = data[testIndex]
        test.append(selected)
        testIndex += 1
        count += 1
    count = 0
    index = testIndex

    #   Adding to train list
    while count < numOfTrain:
        if index == total-1:
            index = 0
            selected = data[index+1]
            train.append(selected)
            index += 1
            count += 1
        else:
            selected = data[index+1]
            train.append(selected)
            index += 1
            count += 1
    count = 0

    """Store class values"""

    #   Add all possible class values to list
    for row in train:
        if row[-1] not in classValues:
            classValues.append(row[-1])

    """Store number of instances of each class"""

    for row in train:
        x = 0
        while row[-1] != classValues[x]:
            x += 1
        if classValues[x] not in classValuesCount:
            classValuesCount.update({classValues[x]:1})
        elif classValues[x] in classValuesCount:
            classCount = classValuesCount.get(classValues[x])
            classValuesCount.update({classValues[x]:classCount+1})

    """Calculate cond. instances according to class & attribute"""
    x = 0
    while x < len(row)-1:   #   Loop attributes
        for row in train:
            if row[x] not in attributeValues:
                if row[x] != '?':
                    attributeValues.append(row[x])
        for value in attributeValues:   #   Values of attributes
            for classValue in classValues:
                counter = 0
                for row in train:
                    if row[x] == value and row[-1] == classValue:
                        counter += 1
                probabilitiesList.append([value, counter, classValue, x]) # Add attribute value, num of instance attribute cond. on class, class, attribute index.
        x += 1

    """Calculate conditional probability"""
    for classCount in probabilitiesList:
        for key in classValuesCount:
            if key == classCount[2]:
                numClass = classValuesCount.get(key)
                condProbability = (classCount[1]+1)/(numClass+len(row))  #   Number of instances of attribute conditioned on class / Total class instances.
        attributeClassProbabilities.append([condProbability, classCount[0], classCount[2], classCount[3]]) # Store probability, attr value, class, attribute index.

    """Predict class"""
    colIndex = 0
    correct = 0
    incorrect = 0

    for row in test:
        for value in classValues:
            conditionalProb = classValuesCount.get(value)/len(train) # Prior class probability
            for probability in attributeClassProbabilities: #   Iterate through probabilities and check each one
                while colIndex < len(row)-1:
                    if row[colIndex] == probability[1] and probability[2] == value and colIndex == probability[3]: #   If same value, class, attribute
                        conditionalProb *= probability[0]
                    colIndex += 1
                colIndex = 0
            classProbabilities.append([value, conditionalProb]) # Add class and probability to list for each class.

        """Get highest probability and class in list"""
        maximum = -1  # Max value
        for value in classProbabilities:
            if value[1] > maximum:
                maximum = value[1]
                if highestValue:  # If list contains item
                    highestValue.pop()
                    highestValue.append([value[0], value[1]])
                else:  # Else list is empty, just add.
                    highestValue.append([value[0], value[1]])  # Class / Probability
        classProbabilities.clear()
        """Check if classified correctly"""
        if row[-1] == highestValue[0][0]:
            correct += 1
        else:
            incorrect += 1

    currentFoldAccuracy = correct / (correct + incorrect)
    print("Current fold accuracy: ", currentFoldAccuracy)
    print("Correct: ", correct)
    print("Incorrect: ", incorrect)
    clearAll()
    totalAcc += currentFoldAccuracy
    k += 1

finalAcc = (totalAcc/(k-1)) # Divide by number of folds -1 since k starts at 1.
print("Total acc", totalAcc)
print("Accuracy: ", finalAcc)

