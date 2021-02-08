# spiral.py
# COMP9444, CSE, UNSW

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.in_hidden = nn.Linear(2, num_hid)
        self.hidden_out = nn.Linear(num_hid, 1)
        """
        #Originally using Sequential
        self.main = nn.Sequential(
            nn.Linear(2, num_hid),
            nn.Tanh(),
            nn.Linear(num_hid, 1),
            nn.Sigmoid()
        )
        """
    def forward(self, input):
        # First we convert the input to polar co-ordinates
        x,y = input[:,0], input[:, 1]
        r,a = torch.sqrt((x**2) + (y**2)), torch.atan2(y,x) # Might need to reshape
        r,a = r.view(r.shape[0], -1), a.view(a.shape[0], -1)
        #Now we have r and a, we need to concatenate into 1
        output = torch.cat((r,a), dim=1)
        #Network:
        self.output_1 = torch.tanh(self.in_hidden(output))
        output = torch.sigmoid(self.hidden_out(self.output_1))
        return output
"""
#For question 6 part c
#Relu
class PolarNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(PolarNet, self).__init__()
        # INSERT CODE HERE
        self.in_hidden = nn.Linear(2, num_hid)
        self.hidden_out = nn.Linear(num_hid, 1)

    def forward(self, input):
        # First we convert the input to polar co-ordinates
        x,y = input[:,0], input[:, 1]
        r,a = torch.sqrt((x**2) + (y**2)), torch.atan2(y,x) # Might need to reshape
        r,a = r.view(r.shape[0], -1), a.view(a.shape[0], -1)
        #Now we have r and a, we need to concatenate into 1
        output = torch.cat((r,a), dim=1)
        #Network:
        self.output_1 = torch.relu(self.in_hidden(output))
        output = torch.sigmoid(self.hidden_out(self.output_1))
        return output
# 1 extra layer
class PolarNet(torch.nn.Module):
  def __init__(self, num_hid):
      super(PolarNet, self).__init__()
      # INSERT CODE HERE
      self.in_hidden = nn.Linear(2, num_hid)
      self.hidden_intermediate = nn.Linear(num_hid, num_hid)
      self.hidden_out = nn.Linear(num_hid, 1)

  def forward(self, input):
      # First we convert the input to polar co-ordinates
      x,y = input[:,0], input[:, 1]
      r,a = torch.sqrt((x**2) + (y**2)), torch.atan2(y,x) # Might need to reshape
      r,a = r.view(r.shape[0], -1), a.view(a.shape[0], -1)
      #Now we have r and a, we need to concatenate into 1
      output = torch.cat((r,a), dim=1)
      #Network:
      self.output_1 = torch.tanh(self.in_hidden(output))
      self.output_2 = torch.tanh(self.hidden_intermediate(self.output_1))
      output = torch.sigmoid(self.hidden_out(self.output_2))
      return output

"""
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.in_hidden1 = nn.Linear(2, num_hid)
        self.hidden1_hidden2 = nn.Linear(num_hid, num_hid)
        self.hidden2_out = nn.Linear(num_hid, 1)
        """
        #Originally using Sequential
        self.main = nn.Sequential(
            #Two fully connect Tanh
            nn.Linear(2, num_hid),
            nn.Tanh(),
            nn.Linear(num_hid, num_hid),
            nn.Tanh(),
            #Linear out with Sigmoid
            nn.Linear(num_hid, 1),
            nn.Sigmoid()
        )
        """

    def forward(self, input):
        self.output_1 = torch.tanh(self.in_hidden1(input))
        self.output_2 = torch.tanh(self.hidden1_hidden2(self.output_1))
        output = torch.sigmoid(self.hidden2_out(self.output_2))
        return output

"""
#For question 6 part c
#relu
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.in_hidden1 = nn.Linear(2, num_hid)
        self.hidden1_hidden2 = nn.Linear(num_hid, num_hid)
        self.hidden2_out = nn.Linear(num_hid, 1)
    def forward(self, input):
        self.output_1 = torch.relu(self.in_hidden1(input))
        self.output_2 = torch.relu(self.hidden1_hidden2(self.output_1))
        output = torch.sigmoid(self.hidden2_out(self.output_2))
        return output

# 1 extra layer
class RawNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(RawNet, self).__init__()
        # INSERT CODE HERE
        self.in_hidden1 = nn.Linear(2, num_hid)
        self.hidden1_hidden2 = nn.Linear(num_hid, num_hid)
        self.hidden2_hidden3 = nn.Linear(num_hid, num_hid)
        self.hidden3_out = nn.Linear(num_hid, 1)
    def forward(self, input):
        self.output_1 = torch.tanh(self.in_hidden1(input))
        self.output_2 = torch.tanh(self.hidden1_hidden2(self.output_1))
        self.output_3 = torch.tanh(self.hidden2_hidden3(self.output_2))
        output = torch.sigmoid(self.hidden3_out(self.output_3))
        return output
"""
"""
#For reference
def graph_output(net):
    xrange = torch.arange(start=-7,end=7.1,step=0.01,dtype=torch.float32)
    yrange = torch.arange(start=-6.6,end=6.7,step=0.01,dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1),ycoord.unsqueeze(1)),1)

    with torch.no_grad(): # suppress updating of gradients
        net.eval()        # toggle batch norm, dropout
        output = net(grid)
        net.train() # toggle batch norm, dropout back again

        pred = (output >= 0.5).float()

        # plot function computed by model
        plt.clf()
        plt.pcolormesh(xrange,yrange,pred.cpu().view(yrange.size()[0],xrange.size()[0]), cmap='Wistia')
"""

def graph_hidden(net, layer, node):
    xrange = torch.arange(start=-7, end=7.1, step=0.01, dtype=torch.float32)
    yrange = torch.arange(start=-6.6, end=6.7, step=0.01, dtype=torch.float32)
    xcoord = xrange.repeat(yrange.size()[0])
    ycoord = torch.repeat_interleave(yrange, xrange.size()[0], dim=0)
    grid = torch.cat((xcoord.unsqueeze(1), ycoord.unsqueeze(1)), 1) #Set up a grid
    with torch.no_grad(): # Temporarily set all the requires_grad flag to false
        net.eval() #Sets the model at evaluation mode
        net(grid) #Using the pre-defined grid
        if layer == 1: #Mainly for Raw - as there is 2 layers
            pred = (net.output_1[:, node]).float()
        elif layer == 2:
            pred = (net.output_2[:, node]).float()
        # plot function computed by model
        plt.pcolormesh(xrange, yrange, pred.cpu().view(yrange.size()[0], xrange.size()[0]), cmap='Wistia')
        #Tmp
        file_name = "hidden_layer_" + str(layer) + "_node_" + str(node) + ".png"
        plt.savefig(file_name)
        #Showing Graphs
        #plt.show()
