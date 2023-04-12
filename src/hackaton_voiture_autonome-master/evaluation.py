import torch
import statistics
from model import get_mobilenet_classif_1, get_classif_lstm_1
from torcheval.metrics import MeanSquaredError, MulticlassAccuracy
from dataloader import get_dataloader, get_classif_dataloader, get_dataloader_LSTM
import math

def evaluate_calsif(model_path):
    model = get_classif_lstm_1()
    model.load_state_dict(torch.load(model_path))
    metric_steer = MulticlassAccuracy(average=None, num_classes=8).to("cuda:0")
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader, validation_dataloader = get_dataloader_LSTM()

    metrics_1 = []
    metrics_2 = []
    metrics_3 = []
    metrics_4 = []
    metrics_5 = []
    metrics_6 = []
    metrics_7 = []
    metrics_8 = []

    val_list_loss = []
    for j, (val_datas_rgb, val_labels_steer) in enumerate(validation_dataloader):
        print(val_datas_rgb.shape)
                        
        # change model mode
        model.eval()

        # moove variables place in memory on GPU RAM
        if torch.cuda.is_available():
            val_datas_rgb = val_datas_rgb.cuda()
            val_labels_steer = val_labels_steer.cuda()
        
        # make prediction
        # outputs_1, outputs_2 = model(val_datas_rgb, val_datas_g)
        outputs = model(val_datas_rgb)

        # compute loss
        loss = criterion(torch.flatten(outputs,0,1), torch.flatten(val_labels_steer,0,1))

        # compute mectric
        metric_steer.update(torch.flatten(outputs, 0,1), torch.flatten(val_labels_steer,0,1))

        # save loss and metric for the current batch
        metrics =  metric_steer.compute()

        metrics_1.append(metrics[0].item())
        metrics_2.append(metrics[1].item())
        metrics_3.append(metrics[2].item())
        metrics_4.append(metrics[3].item())
        metrics_5.append(metrics[4].item())
        metrics_6.append(metrics[5].item())
        metrics_7.append(metrics[6].item())
        metrics_8.append(metrics[7].item())

        metric_steer.reset()

        # save loss and metric for the current batch
        val_list_loss.append(loss.item())

    print("classes metric : ")
    print("classe 0 : " + str(custom_mean(metrics_1)))
    print("classe 1 : " + str(custom_mean(metrics_2)))
    print("classe 2 : " + str(custom_mean(metrics_3)))
    print("classe 3 : " + str(custom_mean(metrics_4)))
    print("classe 4 : " + str(custom_mean(metrics_5)))
    print("classe 5 : " + str(custom_mean(metrics_6)))
    print("classe 6 : " + str(custom_mean(metrics_7)))
    print("classe 7 : " + str(custom_mean(metrics_8)))

    print("loss : " + str(custom_mean(val_list_loss)))

    print("accuracy total : " + str(custom_mean(metrics_1 + metrics_2 + metrics_3 + metrics_4 + metrics_5 + metrics_6 + metrics_7 + metrics_8)))

def custom_mean(L):
    value = None
    count_real_value = 0
    for x in L:
        if not math.isnan(x):
            count_real_value += 1
            if value is not None:
                value += x
            else: 
                value = x
    
    if value is None:
        return math.nan
    else:
        return value/count_real_value
