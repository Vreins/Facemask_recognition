from fastai.vision.all import * #import everthing from vision
import pandas as pd
import timm

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value)
 #cpu vars
    torch.manual_seed(seed_value)
# cpu  vars
    random.seed(seed_value)
 # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
# gpu vars
        torch.backends.cudnn.deterministic = True
 #needed
        torch.backends.cudnn.benchmark = False
#Remember to use num_workers=0 when creating the DataBunch.

device= torch.device("cuda")

def train(model_name='convnext_tiny'):
    train_df = pd.read_csv("train_labels.csv")
    submission = pd.read_csv("SampleSubmission.csv")
    Nosemask = DataBlock(blocks=(ImageBlock, CategoryBlock), 
                         splitter=TrainTestSplitter(0.1, stratify=train_df["target"]),
                         get_x = ColReader(0, pref = "images/images/" ),
                         get_y=ColReader(1),item_tfms = Resize(460), 
                         batch_tfms = aug_transforms(do_flip=True,flip_vert=True,max_lighting=0.4,
                                                     max_zoom=1.2,max_warp=0.2,max_rotate=30,xtra_tfms=None))
    dls = Nosemask.dataloaders(train_df, bs=16, num_workers=0)
    learn = vision_learner(dls,model_name, metrics=[accuracy], path=".") #try convnext_base and convnext_large
    learn.fine_tune(3,cbs=MixUp) #Apply Mixup #increase no of Epochs
    learn.export('model.pkl')

if __name__ == "__main__":
    train()