#!/usr/bin/env python
# coding: utf-8

# ### Data preparation

# This code creates a directory holding an organized version of the original dataset.

# In[1]:


import os, shutil
import numpy as np


# In[2]:


# Path to the full data directory, not categorised into train/val/test sets or category folders
original_dataset_dir = 'data_raw'

# The directory where we will store our dataset, divided into train/val/test directories, and further into category directories 
base_dir = 'data'


# In[3]:


categories = ['alien', 'predator']

# We want to keep our data organized into train and validation folders, each with separate category subfolders
str_train_val = ['train', 'validation']

if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    print('Created directory: ', base_dir)

for dir_type in str_train_val:
    train_test_val_dir = os.path.join(base_dir, dir_type)

    if not os.path.exists(train_test_val_dir):
        os.mkdir(train_test_val_dir)

    for category in categories:
        dir_type_category = os.path.join(train_test_val_dir, category)

        if not os.path.exists(dir_type_category):
            os.mkdir(dir_type_category)
            print('Created directory: ', dir_type_category)


# In[4]:


directories_dict = {}  # To store directory paths for data subsets.

np.random.seed(12)
for cat in categories:
    list_of_images = np.array(os.listdir(os.path.join(original_dataset_dir,cat)))
    print("{}: {} files".format(cat, len(list_of_images)))
    indexes = dict()
    indexes['validation'] = sorted(np.random.choice(len(list_of_images), size=100, replace=False))
    indexes['train'] = list(set(range(len(list_of_images))) - set(indexes['validation']))
    for phase in str_train_val:
        for i, fname in enumerate(list_of_images[indexes[phase]]):
            source = os.path.join(original_dataset_dir, cat, fname)
            destination = os.path.join(base_dir, phase, cat, str(i)+".jpg")
            shutil.copyfile(source, destination)
        print("{}, {}: {} files copied".format(cat, phase, len(indexes[phase])))
        directories_dict[phase + "_" + cat + "_dir"] = os.path.join(base_dir, phase, cat)


# In[6]:


os.listdir()


# In[5]:


directories_dict


# In[6]:


print('Total training alien images:', len(os.listdir(directories_dict['train_alien_dir'])))
print('Total training predator images:', len(os.listdir(directories_dict['train_predator_dir'])))
print("-"*32)
print('Total validation alien images:', len(os.listdir(directories_dict['validation_alien_dir'])))
print('Total validation predator images:', len(os.listdir(directories_dict['validation_predator_dir'])))


# In[ ]:




