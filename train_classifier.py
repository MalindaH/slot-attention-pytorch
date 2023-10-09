import argparse
import datetime
import time
import sys
import torch
from torch import nn
from torchvision.models import resnet50, resnet18, ResNet18_Weights, ResNet50_Weights
from torchvision import models
from tqdm import tqdm

from dataset import *
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_slots', default=7, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--hid_dim', default=64, type=int, help='hidden dimension size')
parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Rate for the learning rate decay.')
parser.add_argument('--decay_steps', default=100000, type=int, help='Number of steps for the learning rate decay.')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--train', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--dataset', default='dermamnist', type=str)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--log', action='store_true')
parser.add_argument('--slot_checkpoint', type=str, help='where to load SA model checkpoint from')

opt = parser.parse_args()

if opt.log:
    model_path = "./models_tmp/"
    model_folder_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"_classifier/"
    log_path = model_path+model_folder_name
    os.mkdir(log_path)
#     f = open(log_path+"config.log", "a")
#     f.write(str(opt)+'\n')
#     f.close()

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(log_path+"logfile.log", "a")
    
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)  

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass    

    sys.stdout = Logger()

print(opt)

class Resnet50Classifier(nn.Module):
    def __init__(self, output_dimension):
        super(Resnet50Classifier, self).__init__()
        # model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # self.model.fc = nn.Linear(self.model.fc.in_features, output_dimension)
        self.extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(model.fc.in_features, output_dimension)
        self.model = nn.Sequential(self.extractor, nn.Flatten(), self.fc)

    def forward(self, x):
        x = self.model(x) # batch_size, output_dimension
        return x

class SlotClassifier(nn.Module):
    def __init__(self, slot_checkpoint, output_dimension, num_slots, num_iterations, hid_dim, resolution):
        super(SlotClassifier, self).__init__()
        self.model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, hid_dim).to(device)
        self.model.load_state_dict(torch.load(slot_checkpoint)['model_state_dict'])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.model.hid_dim * num_slots, output_dimension)
        # self.classifier = nn.Sequential(self.model, nn.Flatten(), self.fc)
        
    def forward(self, x): # x: [128, 3, 28, 28]
        recon_combined, recons, masks, slots = self.model(x) # slots: [128, 7, 64], [batch_size, num_slots, slot_size]
        x = self.flatten(slots) # [128, 448]
        x = self.fc(x)
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def test(model, data_loader, name):
    print(f'==> Evaluating classifier...{name}')
    # if opt.log:
    #     f = open(log_path+"config.log", "a")
    #     f.write(f'==> Evaluating classifier...{name}\n')
        
    model.eval()
    criterion = nn.CrossEntropyLoss()
    softmax = nn.Softmax(dim=-1)

    with torch.no_grad():
        total_loss = 0
        correct = 0
        num_samples = 0
        
        for _, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            labels = labels.squeeze(-1).cuda()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(softmax(outputs), dim=1)
            # preds = torch.argmax(outputs, dim=1)
            assert preds.shape[0] == len(labels)
            correct += torch.sum(preds == labels).item()
            # correct = torch.sum(preds == labels)
            num_samples += len(labels)
            # print(accu, len(labels))
            
        total_loss /= len(data_loader)
        avg_accu = correct/num_samples

        print ("CELoss: {}, Accuracy: {}".format(total_loss, avg_accu))
    #     if opt.log:
    #         f.write("CELoss: {}, Accuracy: {}\n".format(total_loss, avg_accu))
    # if opt.log:
    #     f.close()
    return total_loss


def train_one_epoch(model, dataloader, epoch, i, start, model_name): 
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    total_loss = 0

    for images, labels in tqdm(dataloader):
        images = images.cuda()
        labels = labels.squeeze(-1).cuda()
        i += 1
        if i < opt.warmup_steps:
            learning_rate = opt.learning_rate * (i / opt.warmup_steps)
        else:
            learning_rate = opt.learning_rate
        learning_rate = learning_rate * (opt.decay_rate ** (i / opt.decay_steps))
        optimizer.param_groups[0]['lr'] = learning_rate
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss /= len(train_dataloader)

    tmp_time = datetime.timedelta(seconds=time.time() - start)
    print("Epoch: {}, CELoss: {}, Time: {}".format(epoch, total_loss, tmp_time))
    # if opt.log:
    #     f = open(log_path+"config.log", "a")
    #     f.write("Epoch: {}, CELoss: {}, Time: {}\n".format(epoch, total_loss, tmp_time))
    #     f.close()
    
    return total_loss, i
            
            
def train_classifier(model, model_name, train_dataloader, val_dataloader, train_dataloader_at_eval, test_dataloader):
    print(f'==> Training classifier ...{model_name}')
    # if opt.log:
    #     f = open(log_path+"config.log", "a")
    #     f.write(f'==> Training classifier...{model_name}\n')
    #     f.close()
        
    early_stopper = EarlyStopper(patience=3, min_delta=0.4)
    
    start = time.time()
    i = 0
    for epoch in range(1, opt.num_epochs+1):
        train_loss, i = train_one_epoch(model, train_dataloader, epoch, i, start, model_name)
        if not epoch % 1:
            if opt.log:
                ckpt_name = f'{log_path}{model_name}_epoch{epoch}.ckpt'
                torch.save({'model_state_dict': model.state_dict()}, ckpt_name)
                print(f'Model checkpoint saved as: {ckpt_name}')
                # f = open(log_path+"config.log", "a")
                # f.write(f'Model checkpoint saved as: {ckpt_name}\n')
                # f.close()
            test(model, train_dataloader_at_eval, "train_dataloader_at_eval")
            test(model, test_dataloader, "test_dataloader")
        validation_loss = test(model, val_dataloader, "val_dataloader")
        if early_stopper.early_stop(validation_loss):             
            break



if __name__ == '__main__':
    if opt.dataset.endswith('mnist'):
        resolution = (28, 28)
        loader = MedMNISTDataLoader(opt.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        train_dataloader = loader.get_train_dataloader()
        val_dataloader = loader.get_val_dataloader()
        train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
        test_dataloader = loader.get_test_dataloader()
    elif opt.dataset == "isic2020":
        resolution = (128, 128)
        loader = ISIC2020DataLoader(batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        train_dataloader = loader.get_train_dataloader()
        train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
        test_dataloader = loader.get_test_dataloader()
    elif opt.dataset == "augmented":
        resolution = (28, 28)
        loader = AugmentedMedMNISTDataLoader(opt.dataset_path, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
        train_dataloader = loader.get_train_dataloader()
        train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
        test_dataloader = loader.get_test_dataloader()
    
    model = Resnet50Classifier(loader.n_classes).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    if opt.train:
        train_classifier(model, "Resnet18Classifier", train_dataloader, val_dataloader, train_dataloader_at_eval, test_dataloader)
    if opt.evaluate:
        test(model, train_dataloader_at_eval, "train_dataloader_at_eval")
        test(model, test_dataloader, "test_dataloader")

    # model = SlotClassifier(opt.slot_checkpoint, loader.n_classes, opt.num_slots, opt.num_iterations, opt.hid_dim, resolution)
    # model = torch.nn.DataParallel(model).to(device)
    # if opt.train:
    #     train_classifier(model, "SlotClassifier", train_dataloader)
    # if opt.evaluate:
    #     test(model, train_dataloader_at_eval, "train_dataloader_at_eval")
    #     test(model, test_dataloader, "test_dataloader")
    
