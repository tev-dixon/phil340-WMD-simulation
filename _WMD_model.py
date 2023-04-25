import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv
import subprocess

model_save_path = "WMD_MODEL.pth"
datafile = "data.dat"
influenced_datafile = "influenced_data.dat"

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
def train(net, criterion, optimizer, scheduler, dataset, epochs, base):
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
        if (base + epoch) % 10 == 0:
            print(f"Epoch: {base + epoch + 1}, Loss: {average_loss:.6f}, Eval: {get_eval_percent():.3f}%")

# Prediction method (grab maximum class)
def predict(net, numbers):
    with torch.no_grad():
        input_tensor = torch.cat(numbers, dim=-1).view(1, -1)
        output = net(input_tensor)
        _, predicted = torch.max(output, 1)
        return True if predicted.item() == 1 else False
    
def influence_data():
    with open(influenced_datafile, mode='w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for data in training_data:
            input = (data[0][:2], )
            prediction = predict(loaded_net, input)
            if prediction == 1 and data[1][0] == 1:
                tsv_writer.writerow([data[0][0].tolist()]+[data[0][1].tolist()]+[data[1][0].tolist()]) #report actual
            elif prediction == 0 and data[1][0] == 0:
                tsv_writer.writerow([data[0][0].tolist()]+[data[0][1].tolist()]+[data[1][0].tolist()]) #report actual (by chance)
            elif prediction == 1 and data[1][0] == 0:
                tsv_writer.writerow([data[0][0].tolist()]+[data[0][1].tolist()]+[data[1][0].tolist()]) #report actual
            elif prediction == 0 and data[1][0] == 1:
                tsv_writer.writerow([data[0][0].tolist()]+[data[0][1].tolist()]+[0]) #business failed due to lack of investment, reporting influence error


# MAIN

# Generate fresh data in the 'data.dat' file
process = subprocess.Popen(['python', '_WMD_datagen.py'])

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
epochs=10
base = 0

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(loaded_net.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)  # Learning rate scheduler
train(loaded_net, criterion, optimizer, scheduler, training_data, epochs, base)

for i in range(1, 24):
    base+=10
    # Update the datafile to be influenced after each run
    influence_data()
    # Read in the influenced data data
    data = read_tsv(influenced_datafile)
    # Get the entire dataset as the training dataset
    training_data = [(torch.tensor([float(x[0]), x[1]]).float(), torch.tensor([1 if x[2] else 0])) for x in data]
    # Train the model on the influenced data
    train(loaded_net, criterion, optimizer, scheduler, training_data, 10, base)

# Report testing evaluation
correct = 0
total = len(testing_data)
for data in testing_data:
    input = (data[0][:2], )
    prediction = predict(loaded_net, input)
    if prediction == 1 and data[1][0] == 1:
        #print(f"True positive {input}") #WMD invested and business succeeded
        correct+=1
    elif prediction == 0 and data[1][0] == 0:
        #print(f"True negative {input}") #WMD did not invest and business would have failed
        correct+=1
    #elif prediction == 1 and data[1][0] == 0:
        #print(f"False positive {input}") #WMD invested and business failed
    #elif prediction == 0 and data[1][0] == 1:
        #print(f"False negative {input}") #WMD did not invest and business would have succeed
print(f"WMD MODEL\nCorrect: {correct}/{total} | {((correct/total)*100)}%")

# Save the model's current state
torch.save(loaded_net.state_dict(), model_save_path)
