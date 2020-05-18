import numpy as np
import math

# Reading training data from the csv file.
all_data = np.genfromtxt("feature_extraction-training-data.csv", delimiter=",", encoding="utf-8-sig", 
                              dtype='unicode')
    
# Reading test data from the csv file.
test_data = np.genfromtxt("feature_extraction-test-set.csv", delimiter=",", encoding="utf-8-sig", 
                              dtype='unicode')
# Randomize data before splitting.
np.random.shuffle(all_data)

# FOR DEVLEOPMENT PUSPOSE ONLY - Splitting the available data in a 80/20 split - 80% Training data and 20% Development data.
# training_data, development_data = all_data[:math.floor(0.8*len(all_data))], all_data[math.floor(0.8*len(all_data)):]

training_data = all_data
# Extracting only the 6 features from the training_data.
training_data_features = training_data[:,1:7]

# FOR DEVLEOPMENT PUSPOSE ONLY -  Extracting only the 6 features from the development_data.
# development_data_features = development_data[:,1:7]

# Extracting only the 6 features from the test_data.
test_data_features = test_data[:,1:7]

# Making the values of the last column as 1 since we have considered the bias term as a weight.
training_data_features = np.append(training_data_features,np.ones([len(training_data_features),1]),1)
training_data_features = training_data_features.astype(float)

# FOR DEVLEOPMENT PUSPOSE ONLY 
# development_data_features = np.append(development_data_features,np.ones([len(development_data_features),1]),1)
# development_data_features = development_data_features.astype(float)

test_data_features = np.append(test_data_features,np.ones([len(test_data_features),1]),1)
test_data_features = test_data_features.astype(float)

# Extracting the correct values
correct_values = training_data[:,7]
correct_values = correct_values.astype(float)

# Initializing weights and considering the bias term as the 7th weight.
weights = np.array(np.zeros((7)))

# After playing around with the learning rate, I found that 0.01 gives the best results,
# with the lowest accuracy recored being 81.58% for 1000 iterations of SGD.
learning_rate = 0.01

# Define function to calculate score.
def calculate_score(dot_product):
    return 1/(1+np.exp(-dot_product))
    
# Repeat
for i in range(0,1000):
    # Randomly select a row index with replacement.
    item = np.random.choice(training_data_features.shape[0], None, True)
    
    # Calculate dot product.
    dot_product = np.dot(weights,training_data_features[item])
    
    # Calculate score for selected row.
    score = calculate_score(dot_product)
    
    # Calculate gradient.
    gradient = (score-correct_values[item])*training_data_features[item]

    # Update weights.
    weights -= learning_rate*gradient
    

# Testing on test data.
def predict_outcome(data,data_features):
    final_class_predictions = []
    for i in range(0,len(data)):
        dot_product = np.dot(weights,data_features[i])
        class_predicted = ""
        
        if round(calculate_score(dot_product)) > 0.5:
            class_predicted = "POS"

        else:
            class_predicted = "NEG"
            
        outcome = []
        outcome.append(data[i][0])
        outcome.append(class_predicted)
        final_class_predictions.append(outcome)
    
    return np.array(final_class_predictions)

# FOR DEVLEOPMENT PUSPOSE ONLY 
# final_predictions, accuracy = predict_outcome(development_data, development_data_features)
final_predictions = predict_outcome(test_data, test_data_features)

np.savetxt("output.txt", final_predictions, delimiter='\t',newline='\n', fmt='%s')