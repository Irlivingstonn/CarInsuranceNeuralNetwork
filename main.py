# Name: Isabella Livingston
# Description: Neural Network from Scratch for Car Insurance Claim Prediction

# Importing Assets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt1
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score

# activation function
def sigmoid(x):
    return(1/(1 + np.exp(-x)))

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Creating the Feed forward neural network
def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)    # input from layer 1 
    a1 = relu(z1)  # out put of layer 2 
    z2 = a1.dot(w2)   # input of out layer
    a2 = sigmoid(z2)  # output of out layer
    return(a2)

# initializing the weights randomly
def generate_wt(x, y):
    li =[]
    for i in range(x * y):
        li.append(np.random.randn())
    return(np.array(li).reshape(x, y))
    
# For loss we will be using mean square error(MSE)
# This function calculates the mean squared error between the predicted output and the actual target values.
def loss(out, Y, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones_like(Y)
    
    # Ensure out and Y are numpy arrays
    s = np.square(out - Y)
    if isinstance(Y, (np.ndarray, list)):
        return sample_weight * (np.sum(s)/len(Y))
    return s 

# Back propagation of error 
def back_prop(x, y, w1, w2, alpha, sample_weight):
    # Ensure x and y are numpy arrays
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    
    # Forward pass
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = sigmoid(z2)
    
    # calculating the weighted error in the output layer
    d2 = (a2 - y) * sample_weight
    d1 = d2.dot(w2.T) * (a1 * (1 - a1))

    # Gradient for w1 and w2
    w1_adj = x.T.dot(d1)
    w2_adj = a1.T.dot(d2)
    
    # Updating parameters
    w1 = w1 - alpha * w1_adj
    w2 = w2 - alpha * w2_adj
    
    return(w1, w2)

# Training the model
def train(x, Y, w1, w2, alpha=0.01, epoch=10, class_weight=1.0):
    # Ensure x and Y are numpy arrays
    acc = []
    losss = []
    
    # Create sample weights
    sample_weights = np.where(Y == 1, class_weight, 1.0)
    
    # Training loop
    # For each epoch, we will calculate the loss and update the weights
    # We will also calculate the accuracy and loss for each epoch
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append(loss(out, Y[i], sample_weights[i]))
            w1, w2 = back_prop(x[i], Y[i], w1, w2, alpha, sample_weights[i])
        # print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100) 
        acc.append((1 - (sum(l)/len(x))) * 100)
        losss.append(sum(l)/len(x))
    
    return acc, losss, w1, w2

def generate_wt(x, y):
    li =[]
    for i in range(x * y):
        li.append(np.random.randn())
    return(np.array(li).reshape(x, y))

# Function to evaluate the model
def evaluate(X, y, w1, w2, threshold):
    predictions = []

    # Iterate through each sample in the dataset
    for x in X:
        out = f_forward(x, w1, w2)

        # Apply threshold to the output
        predictions.append(1 if out > threshold else 0)
    
    # Print the classification report
    print("\n=== Detailed Evaluation (threshold = " + str(threshold) + ")===")
    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"Claim Recall: {recall_score(y, predictions, pos_label=1):.4f}")
    print(f"Claim Precision: {precision_score(y, predictions, pos_label=1):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


# Main Function
def main():
    # Loading the testing and training data
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    # Combining the training and testing data for encoding
    combined = pd.concat([df_train, df_test], ignore_index=True)
    
    # Gets all categorical columns
    categorical_cols = [col for col in combined.columns 
                       if combined[col].dtype == 'object' or 
                          combined[col].nunique() < 20]  # Also consider low-cardinality as categorical
    
    # For each column, this sets the string salues (in Dataset) to numerical values
    for col in categorical_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))  
    
    # Seperating the train and test data after encoding
    df_train_clean = combined.iloc[:len(df_train)]
    df_test_clean = combined.iloc[len(df_train):]

    # Dropping these columns since they're the least correlated features (makes the model more accurate)
    cols_to_drop = ['policy_id', 'gear_box', 'transmission_type', 'rear_brakes_type',
                   'is_parking_camera', 'height', 'steering_type', 'max_torque',
                   'population_density', 'age_of_car']
    
    df_train_clean = df_train_clean.drop(cols_to_drop, axis=1)
    df_test_clean = df_test_clean.drop(cols_to_drop, axis=1)

    # Setting up training data, dropping the target variable since it's not needed
    X = df_train_clean.drop('is_claim', axis=1).values
    y = df_train_clean['is_claim'].values
    
    # Splitting dataset into training and testing data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #print("Full dataset class distribution:", np.bincount(y))
    #print("Training set class distribution:", np.bincount(y_train)) 
    #print("Validation set class distribution:", np.bincount(y_val))

    # Normailizing the data and fits the scaler on training data and transforms it 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)         # Applying the same transformation to validation data
    
    # Initializing and training the neural network
    input_size = X_train.shape[1]                 # Number of input features
    hidden_size = 64                              # Number of neurons in the hidden layer
    output_size = 1 
    threshold = 0.3                               # Threshold for classification

    # Generating random weights for the input and hidden layers
    w1 = generate_wt(input_size, hidden_size)  # weights for input -> hidden
    w2 = generate_wt(hidden_size, output_size) # weights for hidden -> output

    # Training the model with different class weights:
    # I noticed that there was an imabalance in the dataset,
    # so I used class weighting to balance the dataset
    for weight in range(15, 26, 1):   # 15:1 ratio
        print("Weight:", weight)

        acc, loss, w1, w2 = train(X_train, y_train, w1, w2, 0.001, 100, weight)
        print("Accuracy:", acc[-1])
        print("Loss:", loss[-1])
        # plotting accuracy
        #plt1.plot(acc)
        #plt1.ylabel('Accuracy')
        #plt1.xlabel("Epochs:")
        #plt1.show()

        # plotting Loss
        #plt1.plot(loss)
        #plt1.ylabel('Loss')
        #plt1.xlabel("Epochs:")
        #plt1.show()

        # After training, evaluating with different thresholds
        evaluate(X_val, y_val, w1, w2, threshold=0.3)  # More sensitive
        evaluate(X_val, y_val, w1, w2, threshold=0.5)  # Original
        evaluate(X_val, y_val, w1, w2, threshold=0.7)  # More conservative to weight change

        # Calculating the F1-Score and ROC-AUC
        y_pred_proba = [f_forward(x, w1, w2) for x in X_val]
        y_pred = [1 if p > 0.5 else 0 for p in y_pred_proba]

        print(f"F1-Score: {f1_score(y_val, y_pred, pos_label=1):.4f}")
        print(f"ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")
        print("--------------------------------------------------")
    print("Program Finished Successfully!")

# Running the main function
if __name__ == "__main__":
    main()