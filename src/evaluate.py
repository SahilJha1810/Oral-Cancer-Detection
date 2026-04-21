import torch
import torch.nn.functional as F

def ensemble_predict(model1, model2, images):
    out1 = model1(images)
    out2 = model2(images)

    p1 = F.softmax(out1, dim=1)
    p2 = F.softmax(out2, dim=1)

    probs = (p1 + p2) / 2
    return probs
