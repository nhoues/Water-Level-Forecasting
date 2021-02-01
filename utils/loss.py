import torch 
import torch.nn as nn 

class EVL() : 
    def __init__(self , alpha , thershold) : 
        self.alpha = alpha 
        self.threshold = threshold 
        self.MSE = nn.MSELoss() 
        
    def __call__(self , y_pred , y_true) :  
        loss = self.MSE(y_pred , y_true) 
        above_threshold = (y_true >= self.thershold).type(torch.float) 
        outliers_loss = self.MSE(
                    y_true[(above_threshold != 0).nonzero(as_tuple=True)],
                    y_pred[(above_threshold != 0).nonzero(as_tuple=True)],
                )
        final_loss = loss + self.alpha * outliers_loss 
        
        return final_loss 