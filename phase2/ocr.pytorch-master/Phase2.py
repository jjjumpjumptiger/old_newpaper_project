#!/usr/bin/env python
# coding: utf-8

# ### Import and Set up

# In[2]:


import numpy as np
import torch
import torchvision
import detectron2
import cv2
import os
import json
import random


# In[3]:


print(torch.__version__, torch.cuda.is_available())


# In[4]:


# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()


# In[5]:


# import some common detectron2 utilities
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# In[6]:


#choose which category to train on
category = "text"


# ### Prepare Dataset (only certain category)
# Convert VOC xml files into a COCO json format file via VOC2COCO package

# In[6]:


register_coco_instances("newspapers",{},"../../VOC2COCO/output_new_"+category+".json","../../21S1_URECA_FYP/ST2020-01")
register_coco_instances("newspapers_test",{},"../../VOC2COCO/output_new_"+category+"_test.json","../../21S1_URECA_FYP/ST2020-01")


# In[7]:


newspapers_metadata = MetadataCatalog.get("newspapers")


# In[8]:


newspapers_dataset_dicts = DatasetCatalog.get("newspapers")


# In[9]:


len(newspapers_dataset_dicts)


# In[10]:


#visualize current pictures
import matplotlib.pyplot as plt


# ### Setting up the model
# We use a Faster RCNN model with a Resnet 50 backbone

# In[11]:


from detectron2.engine import DefaultTrainer

cfg = get_cfg()
config_name = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(config_name))
cfg.DATASETS.TRAIN = ("newspapers",)
cfg.DATASETS.TEST = ("newspapers_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name) 
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 500    
cfg.SOLVER.STEPS = []        
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)


# In[12]:


trainer.train()


# In[13]:


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)


# In[14]:


from detectron2.utils.visualizer import ColorMode
test_dicts = DatasetCatalog.get("newspapers_test")


# In[23]:


from ocr import ocr
import time
import shutil
from glob import glob
from PIL import Image


# In[52]:


def image_to_txt(sorted_boxes):
    t = time.time()
    for i in range(len(sorted_boxes)):
        left = round(sorted_boxes[i]['boundary'][0])
        top = round(sorted_boxes[i]['boundary'][1])
        right = round(sorted_boxes[i]['boundary'][2])
        bottom = round(sorted_boxes[i]['boundary'][3])
        crop_image = image[top:bottom,left:right]
        result, image_framed = ocr(crop_image)
        for key in result:
            f = open('zc_test/'+'IMG_5841_result.txt', "a")
            f.write(result[key][1])
            f.close()
        im = Image.fromarray(image_framed)
        im.save('zc_test/IMG_5841_'+str(i)+'_result.png')
    print("Mission complete, it took {:.3f}s".format(time.time() - t))


# In[53]:


#now only test on the first pic
# for d in test_dicts:
d = test_dicts[0]
#prepare ocr input
image = cv2.imread(d["file_name"])
output = predictor(image)
prediction = output["instances"].to("cpu")
boxes = prediction.pred_boxes if prediction.has("pred_boxes") else None
scores = prediction.scores if prediction.has("scores") else None
boxes_array = boxes.tensor.numpy()
scores_array = scores.numpy()
centers = boxes.get_centers().numpy()/100
unsorted_boxes = []
for i in range(len(boxes_array)):
    if scores_array[i]>= 0.9:
        box = {}
        box['score'] = scores_array[i]
        box['boundary'] = boxes_array[i]
        box['center'] = centers[i]
        unsorted_boxes.append(box)
sorted_boxes = sorted(unsorted_boxes, key=lambda d: (round(d['center'][0]),d['center'][1]))

#OCR
image_to_txt(sorted_boxes)


# In[36]:


# centers = boxes.get_centers().numpy()
# for i in range(len(boxes_array)):
#     if scores_array[i]>= 0.9:
#         box = {}
#         box['score'] = scores_array[i]
#         box['boundary'] = boxes_array[i]
#         box['center'] = centers[i]
#         unsorted_boxes.append(box)
# sorted_boxes = sorted(unsorted_boxes, key=lambda d: (round(d['center'][0]),round(d['center'][1])))

# print(sorted_boxes)


# In[55]:


sorted_boxes


# In[40]:


plt.imshow(image)


# In[56]:


len(unsorted_boxes)


# In[58]:


boxes_array


# In[ ]:




