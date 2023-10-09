import os
import argparse
from dataset import *
from model import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from torchvision import transforms
import torchvision.transforms as T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int, help='random seed')
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
parser.add_argument('--log', action='store_true')
parser.add_argument('--checkpoint', type=str, help='where to load model checkpoint from')

opt = parser.parse_args()
print(opt)

if opt.log:
    model_path = "./models_tmp/"
    model_folder_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+"_generator/"
    log_path = model_path+model_folder_name
    os.mkdir(log_path)
    f = open(log_path+"config.log", "a")
    f.write(str(opt)+'\n')
    f.close()

if opt.dataset.endswith('mnist'):
    resolution = (28, 28)
    loader = MedMNISTDataLoader(opt.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader = loader.get_train_dataloader()
    train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
    test_dataloader = loader.get_test_dataloader()
elif opt.dataset == "isic2020":
    resolution = (128, 128)
    loader = ISIC2020DataLoader(batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader = loader.get_train_dataloader()
    train_dataloader_at_eval = loader.get_train_dataloader_at_eval()
    test_dataloader = loader.get_test_dataloader()
    
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0.],
                                                     std = [ 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5 ],
                                                     std = [ 1. ])])

model = SlotAttentionAutoEncoder(resolution, opt.num_slots, opt.num_iterations, opt.hid_dim).to(device)
if opt.checkpoint:
    model.load_state_dict(torch.load(opt.checkpoint)['model_state_dict'])

## evaluation:
def test(split):
    print('==> Evaluating ...')
    
    if opt.log:
        f = open(log_path+"config.log", "a")
        f.write('==> Evaluating ...\n')
        
        img_folder = log_path + "train/" if split == 'train' else log_path + "test/" 
        os.mkdir(img_folder)
        metadata = open(img_folder+"metadata.csv", "a")
        metadata.write('image_name,labels\n')
        to_pil = T.ToPILImage()
        
    model.eval()
    criterion = nn.MSELoss()
    data_loader = train_dataloader_at_eval if split == 'train' else test_dataloader

    with torch.no_grad():
        total_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            images = images.cuda()
            
            recon_combined, recons, masks, slots = model(images)
            loss = criterion(recon_combined, images)
            total_loss += loss.item()
            
            if opt.log:
                for j in range(len(images)):
                    # plt.imshow(invTrans(np.transpose(images[j].cpu(), (1, 2, 0))))
                    # plt.axis('off')
                    # plt.savefig(img_folder+f'img_{i}_{j}_original.png', dpi=300)
                    # plt.imshow(invTrans(np.transpose(recon_combined[j].cpu(), (1, 2, 0))))
                    # plt.axis('off')
                    # plt.savefig(img_folder+f'img_{i}_{j}_generated.png', dpi=300)
                    
                    img = to_pil(invTrans(images[j]))
                    img.save(img_folder+f'img_{i}_{j}_original.png')
                    img = to_pil(invTrans(recon_combined[j]))
                    img.save(img_folder+f'img_{i}_{j}_generated.png')
                    metadata.write(f'img_{i}_{j}_original.png,{targets[i][0]}\n')
                    if split == 'train':
                        metadata.write(f'img_{i}_{j}_generated.png,{targets[i][0]}\n')

            del recons, masks, slots
            
        total_loss /= len(data_loader)

        print ("{}: MSELoss: {}".format(split, total_loss))
        if opt.log:
            f.write("{}: MSELoss: {}\n".format(split, total_loss))
    if opt.log:
        f.close()
        metadata.close()
            

## training:
if opt.train:
    print('==> Training ...')
    
    if opt.log:
        f = open(log_path+"config.log", "a")
        f.write('==> Training ...\n')
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    criterion = nn.MSELoss()

    start = time.time()
    i = 0
    for epoch in range(1, opt.num_epochs+1):
        model.train()

        total_loss = 0

        for images, targets in tqdm(train_dataloader):
            images = images.cuda()
            
            i += 1

            if i < opt.warmup_steps:
                learning_rate = opt.learning_rate * (i / opt.warmup_steps)
            else:
                learning_rate = opt.learning_rate

            learning_rate = learning_rate * (opt.decay_rate ** (
                i / opt.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            recon_combined, recons, masks, slots = model(images)
            loss = criterion(recon_combined, images)
            total_loss += loss.item()

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss /= len(train_dataloader)

        tmp_time = datetime.timedelta(seconds=time.time() - start)
        print("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss, tmp_time))
        if opt.log:
            f.write("Epoch: {}, Loss: {}, Time: {}\n".format(epoch, total_loss, tmp_time))
        if not epoch % 10:
            if opt.log:
                ckpt_name = f'{log_path}SAAutoEncoder_epoch{epoch}.ckpt'
                torch.save({'model_state_dict': model.state_dict()}, ckpt_name)
                print(f'Model checkpoint saved as: {ckpt_name}')
                f.write(f'Model checkpoint saved as: {ckpt_name}\n')
            test('train')
            test('test')
    if opt.log:
        f.close()


if opt.evaluate:
    test('train')
    test('test')