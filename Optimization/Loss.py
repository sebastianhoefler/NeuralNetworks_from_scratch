import numpy as np

class CrossEntropyLoss():
    def __init__(self) -> None:
        pass


    def forward(self, prediction_tensor,label_tensor):
        # save for backward pass
        self.prediction_tensor = prediction_tensor

        # get machine percision
        eps = np.finfo(float).eps

        # get entries where label is 1
        label_indices = np.nonzero(label_tensor == 1)

        #return values of indices in prediction_tensor
        prediction_values = self.prediction_tensor[label_indices]

        # take -ln(y_hat+eps) over entries
        ce_rows = -np.log(prediction_values + eps)

        # sum over all rows to get total loss over all batches
        ce_loss = np.sum(ce_rows) 
        
        return ce_loss

    def backward(self, label_tensor):
        # get machine percision
        eps = np.finfo(float).eps

        error_tensor = - label_tensor / (self.prediction_tensor + eps)

        return error_tensor