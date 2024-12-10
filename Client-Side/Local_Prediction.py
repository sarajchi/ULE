if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping
root_directory = 'Temporary_Data'                   # Directory where temporary folders are stored
time_for_prediction = 25                            # Time we wait for each prediction
prediction_threshold = 3                            # how much prediction we need to activate
# ------------------------------------

import os
import sys
import time

try :
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '')
    missing_module = missing_module.replace("'", '')
    sys.exit(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.InferenceDataloader import HAR_Inference_DataSet
    from Imports.Functions import model_exist
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

def make_prediction(Dataset) :
    Loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in Loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
            predicted = torch.argmax(model(video_frames, imu_data))
    return predicted

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

# If there is no model to load, we stop
if not model_exist() :
    sys.exit("No model to load") # If there is no model to load, we stop


try :
    Done = False
    while not Done :
        try :
            if len(os.listdir(root_directory)) > 1 :
                Done = True
            else : time.sleep (0.1)
        except FileNotFoundError : 
            pass
        print('Waiting for data, launch GetData.py')
        time.sleep(0.1)
        print(LINE_UP, end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

'''input('Programme Ready, Press Enter to Start')
for i in range(3) :
    print(f'Starting in {3-i}s')
    time.sleep(1)
    print(LINE_UP, end=LINE_CLEAR)'''

Start_Tracking_Time = time.time()

idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)

ModelToLoad_Path = os.path.join('Model to Load',os.listdir('./Model to Load')[0])
ModelName = os.listdir('./Model to Load')[0]
if ModelName.endswith('.pt') :
    ModelName = ModelName.replace('.pt','')
else :
    ModelName = ModelName.replace('.pht','')
print(f"Loading {ModelName}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

model = FusionModel(config.MODEL.MoViNetA0, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

try : # Main Loop
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n')
    old_sample = ''
    first_sample = ''
    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action
    if not os.listdir(root_directory) :
        print('No files in root directory')
        sys.exit(0)
    while True:
        while old_sample == dataset.SampleNumber :
            time.sleep(0.001)
            dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
        old_sample = dataset.SampleNumber

        try :
            prediction = make_prediction(dataset)
        except :
            print(f'Error on {old_sample}')
        tracking[prediction] += 1
        print(f'{old_sample} : {idx_to_action.get(prediction.item())} at {round(time.time()-Start_Tracking_Time,2)}')
        if first_sample == '' : first_sample = old_sample


except KeyboardInterrupt:
    pass

except FileNotFoundError:
    print("Samples folder got deleted")
    
num_of_predictions = 0
for i in tracking :
    num_of_predictions += i
num_first = int(first_sample.replace('Sample_',''))
num_last = int(old_sample.replace('Sample_',''))

if num_of_predictions > 1 : end_text = 's'
else : end_text = ''
print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, with {(num_last-num_first+1)-num_of_predictions} missed')
for action, i in action_to_idx.items() :
    print(f'{tracking[i]} for {action}')



if __name__ == "__main__" :
    print('\nProgramme Stopped\n')
