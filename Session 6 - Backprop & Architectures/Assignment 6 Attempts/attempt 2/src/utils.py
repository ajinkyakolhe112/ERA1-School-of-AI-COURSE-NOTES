import torch
import torch.nn as nn
import torch.optim as optim
from data import *  # or import data as data
from model import *
from tqdm import tqdm  # just wrap any iterable with tqdm(iterable), and you're done!

trainDataLoader, testDataLoader
model = base_mark1()
errorFun = nn.functional.nll_loss

epochs = 5
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr)
# device = torch.device("mps")


def train(trainDataLoader, model, errorFun, optimizer, device=None):
    pbar = tqdm(trainDataLoader)
    batch_size = trainDataLoader.batch_size
    for epoch in range(epochs):
        # print("epoch=",epoch)
        errorValueTotal = 0
        correctTotal = 0
        totalProcessed = 0
        for batchId, (Xdata, Yobserved) in enumerate(pbar):
            batchId

            "Y = X \cdot W "
            Ypred = model(Xdata)
            errorValue = errorFun(Ypred, Yobserved, reduction="mean")
            errorValue.backward()
            # error for Human Reading
            predictMaxProbClassIndex = Ypred.argmax(dim=1)
            comparison = Yobserved.eq(predictMaxProbClassIndex)
            correctPredictions = comparison.sum().item()

            "W = W - gradient * lr "
            optimizer.step()
            # 1. Calculate how many weights are being changed. (tensor[tensor != 0], tensor[tensor.nonzero()])
            # 2. weight values are not copied in memory for tracking

            "Reset Graph"
            optimizer.zero_grad()

            pbar.set_description("Batch No: %d, \tError: %f,\t Correct: %d,\t"
                % (batchId, errorValue.item(), correctPredictions))
            errorValueTotal += errorValue.item()
            correctTotal += correctPredictions
            totalProcessed += Xdata.shape[0]
            if batchId % 10 == 0:
                pass
        print(errorValueTotal / totalProcessed, 100 * (correctTotal / totalProcessed))


def test(testDataLoader, model, errorFun, device=None):
    testLossTotal = 0
    correctPredsTotal = 0
    totalProcessed = 0
    for batchId, (Xdata, Yobserved) in enumerate(testDataLoader):
        Ypred = model(Xdata)
        testLoss = errorFun(Ypred, Yobserved, reduction="mean").item()
        correctPreds = torch.eq(Ypred.argmax(dim=1), Yobserved).sum().item()

        testLossTotal += testLoss
        correctPredsTotal += correctPreds
        totalProcessed += Xdata.shape[0]
    print(testLossTotal)
    print(100 * (correctPredsTotal / totalProcessed))


if __name__ == "__main__":
    "Accessing weights of any GENERAL model"
    model.conv_block1[0].weight
    model.conv_block1[3].weight

    train(trainDataLoader, model, errorFun, optimizer)
    test(testDataLoader, model, errorFun)


print("END")
