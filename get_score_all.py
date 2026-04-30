
all_name_list = ['vase', 'rugby_ball', 'starfish', 'scuba_diver', 'italian_greyhound', 'espresso', 'broccoli', 'cauldron', 'cannon', 'steam_locomotive', 'zebra', 'scarf', 'Granny_Smith', 'tiger', 'panda', 'collie', 'hotdog', 'standard_poodle', 'tennis_ball', 'canoe', 'goldfinch', 'baseball_player', 'yorkshire_terrier', 'stingray', 'gorilla', 'trombone', 'pig', 'hummingbird', 'accordion', 'monarch_butterfly', 'submarine', 'mailbox', 'scottish_terrier', 'bucket', 'eel', 'hyena', 'afghan_hound', 'tractor', 'sea_lion', 'shih_tzu', 'great_white_shark', 'bloodhound', 'school_bus', 'husky', 'bee', 'orangutan', 'timber_wolf', 'puffer_fish', 'baboon', 'pelican', 'flamingo', 'ladybug', 'polar_bear', 'bathtub', 'mitten', 'cocker_spaniels', 'centipede', 'mushroom', 'carousel', 'bagel', 'newt', 'guillotine', 'harp', 'pembroke_welsh_corgi', 'whippet', 'binoculars', 'dalmatian', 'beer_glass', 'boxer', 'backpack', 'grand_piano', 'meerkat', 'missile', 'hermit_crab', 'cowboy_hat', 'ice_cream', 'hammer', 'king_penguin', 'space_shuttle', 'volcano', 'skunk', 'pug', 'axolotl', 'African_chameleon', 'dragonfly', 'red_fox', 'beagle', 'cucumber', 'black_swan', 'junco', 'hen', 'jeep', 'bison', 'birdhouse', 'hatchet', 'strawberry', 'grasshopper', 'grey_whale', 'pineapple', 'boston_terrier', 'burrito', 'saxophone', 'pretzel', 'ostrich', 'lorikeet', 'beaver', 'ant', 'fly', 'guinea_pig', 'gazelle', 'chow_chow', 'bell_pepper', 'labrador_retriever', 'koala', 'fire_engine', 'violin', 'flute', 'chihuahua', 'pirate_ship', 'french_bulldog', 'spider_web', 'banana', 'lawn_mower', 'tree_frog', 'sandal', 'hippopotamus', 'jellyfish', 'cheeseburger', 'electric_guitar', 'toy_poodle', 'bald_eagle', 'lion', 'clown_fish', 'castle', 'candle', 'pomeranian', 'pomegranate', 'chimpanzee', 'parachute', 'rottweiler', 'lemon', 'badger', 'harmonica', 'snow_leopard', 'cabbage', 'iguana', 'wine_bottle', 'mantis', 'military_aircraft', 'cockroach', 'soccer_ball', 'leopard', 'german_shepherd_dog', 'assault_rifle', 'duck', 'west_highland_white_terrier', 'wheelbarrow', 'joystick', 'cobra', 'cheetah', 'scorpion', 'lighthouse', 'killer_whale', 'fox_squirrel', 'pizza', 'golden_retriever', 'saint_bernard', 'lipstick', 'revolver', 'basketball', 'american_egret', 'acorn', 'peacock', 'ambulance', 'toucan', 'lab_coat', 'goldfish', 'barn', 'pickup_truck', 'broom', 'mobile_phone', 'snail', 'border_collie', 'bow_tie', 'hammerhead', 'vulture', 'tabby_cat', 'goose', 'llama', 'shield', 'tarantula', 'schooner', 'tank', 'porcupine', 'gibbon', 'weimaraner', 'basset_hound', 'wood_rabbit', 'lobster', 'gasmask']




# ImageNet-R B100P10
path_results = ''

initial = 20
increment = 20
task_num = 10

# original
import torch
import clip
from PIL import Image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

all_name = os.listdir(path_results)
all_name.sort()
all_name = sorted(all_name, key=lambda x: int(x.split('.')[0][5:]))
all_mean  = []

for task_id, name in enumerate(all_name):
    with open(path_results + name) as f:
        label_list = []
        msg_list = []
        lines = f.readlines()
        inner_task = [[] for i in range(task_num)]
        idx = 0
        for line in lines:
            if line.startswith('the label is'):
                line = line[13:]
                line = line.strip('\n')
                for tasks in range(task_id + 1):
                    if tasks == 0:
                        if line in all_name_list[:initial]:
                            inner_task[tasks].append(idx)
                    elif line in all_name_list[initial + (tasks-1)*increment: initial+tasks*increment]:
                        inner_task[tasks].append(idx)
                idx += 1             
                label_list.append(line)
            if line.startswith('msg:'):
                line = line[26:]
                line = line.strip('\n')
                line = line.strip('.')
                line = line.strip('#')
                msg_list.append(line) 
        label_set = list(set(label_list))
        text = clip.tokenize(label_set).to(device)

        new_msg_list = []
        for mm in msg_list:
            if len(mm) > 76:
                new_msg_list.append(mm[:76])
            else:
                new_msg_list.append(mm)

        msg_list = new_msg_list

        predict_text = clip.tokenize(msg_list).to(device)

        with torch.no_grad():  
            text_features_label = model.encode_text(text)
            predict_feature = model.encode_text(predict_text)

            text_features_label = text_features_label / text_features_label.norm(dim=1, keepdim=True)
            predict_feature = predict_feature / predict_feature.norm(dim=1, keepdim=True)

        sim  = predict_feature @ text_features_label.T
        real_label = torch.ones(len(msg_list))
        for i in range(len(msg_list)):
            real_label[i] = label_set.index(label_list[i])

        try:
            pre = sim.cpu().argmax(dim=1)
        except:
            import pdb; pdb.set_trace()

        acc = sum(pre == real_label) / len(msg_list)
        task_acc = []
        
        for i in range(task_id + 1):
            task_acc.append(sum(pre[inner_task[i]] == real_label[inner_task[i]])/len(inner_task[i]))

        for acc_each in task_acc:
            print(str(round(float(acc_each*100), 2)), end=' ')

        print( 'mean: ', str(round(float(acc*100), 2)))

        mean = round(float(acc*100), 2)
        all_mean.append(mean)
print('avg: ', sum(all_mean)/len(all_mean))


# # new
# import torch
# import clip
# from PIL import Image
# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)

# all_name = os.listdir(path_results)
# all_name.sort()
# all_name = sorted(all_name, key=lambda x: int(x.split('.')[0][5:]))
# all_mean  = []

# batch_size = 512  # Set your desired batch size

# for task_id, name in enumerate(all_name):
#     with open(path_results + name) as f:
#         label_list = []
#         msg_list = []
#         lines = f.readlines()
#         inner_task = [[] for i in range(task_num)]
#         idx = 0
#         for line in lines:
#             if line.startswith('the label is'):
#                 line = line[13:]
#                 line = line.strip('\n')
#                 for tasks in range(task_id + 1):
#                     if tasks == 0:
#                         if line in all_name_list[:initial]:
#                             inner_task[tasks].append(idx)
#                     elif line in all_name_list[initial + (tasks-1)*increment: initial+tasks*increment]:
#                         inner_task[tasks].append(idx)
#                 idx += 1             
#                 label_list.append(line)
#             if line.startswith('msg:'):
#                 line = line[26:]
#                 line = line.strip('\n')
#                 line = line.strip('.')
#                 line = line.strip('#')
#                 msg_list.append(line) 

#         label_set = list(set(label_list))
#         text = clip.tokenize(label_set).to(device)

#         new_msg_list = []
#         for mm in msg_list:
#             if len(mm) > 76:
#                 new_msg_list.append(mm[:76])
#             else:
#                 new_msg_list.append(mm)

#         msg_list = new_msg_list

#         num_batches = len(msg_list) // batch_size + (len(msg_list) % batch_size != 0)
#         msg_batches = [msg_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
#         label_batches = [label_list[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

#         all_accs = []
#         task_accs = [[] for _ in range(task_id + 1)]  # Collect accuracy for each sub-task
#         for batch_idx, (msg_batch, label_batch) in enumerate(zip(msg_batches, label_batches)):
#             predict_text = clip.tokenize(msg_batch).to(device)

#             with torch.no_grad():  
#                 text_features_label = model.encode_text(text)
#                 predict_feature = model.encode_text(predict_text)

#                 text_features_label = text_features_label / text_features_label.norm(dim=1, keepdim=True)
#                 predict_feature = predict_feature / predict_feature.norm(dim=1, keepdim=True)

#             sim  = predict_feature @ text_features_label.T
#             real_label = torch.ones(len(label_batch))
#             for i in range(len(label_batch)):
#                 real_label[i] = label_set.index(label_batch[i])

#             pre = sim.cpu().argmax(dim=1)
#             acc = sum(pre == real_label) / len(label_batch)
#             all_accs.append(acc)

#             # Calculate accuracy for each sub-task
#             for i in range(task_id + 1):
#                 # Find the indices for this sub-task in the current batch
#                 batch_start = batch_idx * batch_size
#                 batch_end = batch_start + len(msg_batch)
#                 sub_task_indices = [idx for idx in inner_task[i] if batch_start <= idx < batch_end]

#                 # Adjust the indices for the current batch
#                 sub_task_indices = [idx - batch_start for idx in sub_task_indices]

#                 if sub_task_indices:  # If this sub-task has data in the current batch
#                     sub_task_pre = pre[sub_task_indices]
#                     sub_task_real_label = real_label[sub_task_indices]
#                     task_acc = sum(sub_task_pre == sub_task_real_label) / len(sub_task_indices)
#                     task_accs[i].append(task_acc)  # Add this sub-task's accuracy to the list

#         mean_acc = sum(all_accs) / len(all_accs)
#         print( 'mean: ', str(round(float(mean_acc*100), 2)))

#         # Print the average accuracy for each sub-task
#         for i, task_acc in enumerate(task_accs):
#             avg_task_acc = (sum(task_acc) / len(task_acc)) * 100
#             print(f'task {i}: {avg_task_acc:.2f}', end=' ')
#         print()

#         all_mean.append(mean_acc)

# print('avg: ', sum(all_mean)/len(all_mean))
