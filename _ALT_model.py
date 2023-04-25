import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import subprocess

model_save_path = "ALT_model.pth"
datafile = "data.dat"

# Read in data from datagen
def read_tsv(filename):
    with open(filename, 'r') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        rows = []
        for row in reader:
            row = [float(element) for element in row]
            rows.append(row)
        return rows

# Define a feedforward neural network class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 64) # 2 inputs
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 2) # outputs the probability of the two output classes (invest/don't invest)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# Load the previous model
def load_model(model_save_path):
    net = SimpleNet()
    net.load_state_dict(torch.load(model_save_path))
    net.eval()  # set model to evaluation mode
    return net

def get_eval_percent():
    correct = 0
    total = len(testing_data)
    for data in testing_data:
        input = (data[0][:2], )
        prediction = predict(loaded_net, input)
        if prediction == data[1][0]:
            correct+=1
    return correct/total*100
    

# Train and report loss
def train(net, criterion, optimizer, scheduler, dataset, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in dataset:
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output.view(1, -1), target.view(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        average_loss = epoch_loss / len(dataset)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch + 1}, Loss: {average_loss:.6f}, Eval: {get_eval_percent():.3f}%")
            

# Prediction method (grab maximum class)
def predict(net, numbers):
    with torch.no_grad():
        input_tensor = torch.cat(numbers, dim=-1).view(1, -1)
        output = net(input_tensor)
        _, predicted = torch.max(output, 1)
        return True if predicted.item() == 1 else False


# MAIN

# Generate fresh data in the 'data.dat' file
process = subprocess.Popen(['python', '_ALT_datagen.py'])

# Wait for the process to finish
process.wait()

# Read in the fresh data
data = read_tsv(datafile)

# Get the entire dataset
dataset = [(torch.tensor([float(x[0]), x[1]]).float(), torch.tensor([1 if x[2] else 0])) for x in data]

# Splinter into 90% data, 10% testing
n = len(dataset)
training_data = dataset[:int(n*0.9)]
testing_data = dataset[int(n*0.9):]

# Create or load model
loaded_net = SimpleNet() # uncomment for a new model
#loaded_net = load_model(model_save_path) # uncomment to use previous model

# Set Hyperparams
learning_rate = 0.004
step_size=8
gamma=0.9
epochs=250

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(loaded_net.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)  # Learning rate scheduler
train(loaded_net, criterion, optimizer, scheduler, training_data, epochs)

# Report testing evaluation
correct = 0
total = len(testing_data)
for data in testing_data:
    input = (data[0][:2], )
    prediction = predict(loaded_net, input)
    if prediction == 1 and data[1][0] == 1:
        #print(f"True positive {input}")
        correct+=1
    elif prediction == 0 and data[1][0] == 0:
        #print(f"True negative {input}")
        correct+=1
    #elif prediction == 1 and data[1][0] == 0:
        #print(f"False positive {input}")
    #elif prediction == 0 and data[1][0] == 1:
        #print(f"False negative {input}")
print(f"ALT MODEL:\nCorrect: {correct}/{total} | {((correct/total)*100)}%")

# Save the model's current state
torch.save(loaded_net.state_dict(), model_save_path)
