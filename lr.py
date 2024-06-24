import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax

#linear regression using sklearn model 
'''from sklearn.linear_model import LinearRegression
lr_model=LinearRegression()
lr_model.fit(train_input,train_output)
lr_model.coef_
lr_model.intercept_
test_predictions=lr_model.predict(test_input)
plt.plot(test_input,test_predictions,"-"",color="blue"
#cost function 
from sklearn.metrices import mean_squared_error
cost=mean_squared_error(test_output,test_predictions)

#fit is used for the training and it has already built in function inside which we will create
'''

#creating lr model from scratch 


parameters = {
    "m": np.random.uniform(0, 2),
    "c": np.random.uniform(0, 1)
}


data = pd.read_csv(r"C:\Users\KIIT\Downloads\areas.csv")

print(data.head(10))
print(data.isnull().sum())
data=data.dropna()
print("shape of the data:",data.shape)
print(data.info())

#training input and ouput
train_input = np.array(data['area'][0:5]).reshape(5, 1)
train_output = np.array(data['price'][0:5]).reshape(5, 1)


#validation input and output

test_input = np.array(data['area'][5:10]).reshape(5, 1)
test_output = np.array(data['price'][5:10]).reshape(5, 1)
print(f"training input data shape={train_input.shape}")
print(f"testing data shape={test_input.shape}")
print(f"training output data shape={train_output.shape}")
print(f"testing output data shape={test_output.shape}")

#linear regression
#forward propogation

def forward_propagation(train_input, parameters):

    m=parameters["m"]
    c=parameters["c"] #here parameters are dictionary

    predictions = np.multiply(m, train_input) +c
    return predictions


#cost function summation(1/2n*(y-f(x)^2)

def cost_function(predictions, train_output):
    cost = np.mean((train_output - predictions) ** 2) * 0.5
    return cost


#gradient descent or backward propogation
#df=(f(x)-y)/n here f(x) is the predicted y which is the result of forward propogation
#dm=df*x
#dc=df*1

def backward_propagation(train_input, train_output, predictions):
    derivatives = dict()

    df = np.mean(predictions - train_output)
    dm = np.multiply(df, train_input)
    dc = df
    derivatives["dm"] = dm #derivatives is the dictionary holdigf keys dm and dc
    derivatives["dc"] = dc
    return derivatives


# #update parameters
# m=m-(learning rate*dm)
# c=c-(learning rate*dc)


def update_parameters(parameters, derivatives, learning_rate):
    parameters["m"] = parameters["m"] - learning_rate * derivatives["dm"]
    parameters["c"] = parameters["c"] - learning_rate * derivatives["dc"]
    return parameters

def train(train_input, train_output, learning_rate, iters):
    parameters = dict()
    parameters["m"] = np.random.uniform(0, 1)
    parameters["c"] = np.random.uniform(0, 1)

    loss = []

    for i in range(iters):
        predictions = forward_propagation(train_input, parameters)
        cost = cost_function(predictions, train_output)
        loss.append(cost)
        print(f"Iteration {i+1}, loss {cost}")

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(train_input, train_output, color='blue', label='Original Data')
        plt.plot(train_input, predictions, color='red', label='Predicted Data')
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Backward propagation
        derivatives = backward_propagation(train_input, train_output, predictions)

        # Update parameters
        parameters = update_parameters(parameters, derivatives, learning_rate)

    # Return outside the loop to ensure all iterations are completed
    return parameters, loss

parameters, loss = train(train_input, train_output, 0.001, 10)
print(parameters)
plt.plot(loss)

test_predictions=test_input*parameters['m']+parameters['c']
plt.plot(test_predictions,test_predictions,"^")
plt.plot(test_input,test_output,",")
plt.show()#this is the test inout which our model has never seen and is a fresh value for the model 
print(cost_function(test_input,test_predictions))