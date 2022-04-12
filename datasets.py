import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from vit import create_vit
from cvt import create_cvt

class MyDataset(Dataset):
    def __init__(self,image_path,name2id):
        super().__init__()
        self.image_list = os.listdir(image_path)
        self.image_path = image_path
        self.name2id = name2id
    
    def __getitem__(self, index):
        transform=transforms.Compose([transforms.Resize((384,384)),
                                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        image_file = self.image_path+"/"+self.image_list[index]
        image = Image.open(image_file).convert("RGB")
        img = transform(image)
        image.close()
        id = self.name2id[self.image_list[index]]
        id = torch.tensor(id,dtype=torch.long)
        return img,id
        
    
    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from datasets import MyDataset
    from torch.utils.data import DataLoader
    from progress.bar import ShadyBar
    import warnings
    warnings.filterwarnings("ignore")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f = open("I:\\imagenet_val\\val_list.txt")
    name2id={}
    line = f.readline()
    while line:
        name,id=line.split("\n")[0].split(' ')[0],int(line.split("\n")[0].split(' ')[1])
        name2id[name]=id
        line=f.readline()
    f.close()
    model = create_cvt("CvT-13-224x224-IN-1k").to(device)
    # model = create_vit("vit_large_patch16_384").to(device)
    model.eval()
    #"I:\\ilsvrc2012_img_val"为imagenet1k验证集数据地址
    datasets = MyDataset("I:\\imagenet_val\\ILSVRC2012_img_val",name2id)
    dataloader = DataLoader(datasets,48,False,num_workers=8)
    y_true = []
    y_pred = []
    with torch.no_grad():
        bar = ShadyBar("valid",max = len(dataloader))
        for i,datas in enumerate(dataloader):
            inputs,labels = datas
            inputs,labels = inputs.to(device),labels.numpy()
            out = model(inputs)
            _,pred = out.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            bar.next()
        bar.finish()
        print("acc:{}".format(accuracy_score(y_true,y_pred)))