import torch.nn as nn
import torch.nn.functional as F 
import torch
import dataset
from sklearn.metrics import accuracy_score

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
                

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5) 
        # self.dropout3=nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 7)  
   
    def forward(self,x):
        x=F.relu(self.conv1(x)) 
        x=F.max_pool2d(x,2)  
        x=F.relu(self.conv2(x)) 
        x=F.max_pool2d(x,2) 
        x=F.relu(self.conv3(x)) 
        x=F.max_pool2d(x,2) 
        x=x.view(-1,128*4*4) 
        x=self.dropout1(x)
        x = F.relu(self.fc1(x))
        x=self.dropout2(x) 
        # x=self.dropout3(x)/
        return self.fc2(x)


epochs=10
learning_rate=0.001
model=EmotionCNN()  
criterion=nn.CrossEntropyLoss() 
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4) 



model.train()
for epoch in range(epochs):
    total_epoch_loss = 0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in dataset.train_loader:
        optimizer.zero_grad()
        output = model(batch_features)
        
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_epoch_loss += loss.item()
        
        
        _, predicted = torch.max(output.data, 1)  
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    avg_loss = total_epoch_loss / len(dataset.train_loader)
    accuracy = 100 * correct / total
    
    print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

model.eval()
total=0 
correct=0 
with torch.no_grad():
    for batch_features,batch_labels in dataset.test_loader: 
        output=model(batch_features) 
        _,predicted=torch.max(output,1) 
        total=total+batch_labels.shape[0]  
        correct=correct+(predicted==batch_labels).sum().item()
print(correct/total) 
