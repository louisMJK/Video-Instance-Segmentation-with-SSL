from simvp_model import SimVP_Model


class LargeModel(nn.Module):
    """Large Model for the End to End Task"""

    def __init__(self):
        super(LargeModel, self).__init__()
        self.predictor = SimVP_Model()



    def forward(self, x, **kwargs):
        x = self.predictor(x)
        #TODO: add the rest of the model

        return x