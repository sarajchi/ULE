if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping
root_directory = 'Temporary_Data'                   # Directory where temporary folders are stored
prediction_threshold = 3                            # how much prediction we need to activate
STOP_ALL = True                                    # If true stops GetData.py as well (You won't get the stats of GetData.py if set to True)
# ------------------------------------

#TODO testing to see if it works
#TODO don't forget to set glaze frequency back to 200Hz

import os
import sys
import time
import socket

try :
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import dotenv
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '')
    missing_module = missing_module.replace("'", '')
    if missing_module == 'dotenv' :
        sys.exit(f'No module named {missing_module} try : pip install python-dotenv')
    else : sys.exit(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.InferenceDataloader import HAR_Inference_DataSet
    from Imports.Functions import model_exist, all_the_same
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'
dotenv.load_dotenv()

bufferSize = 1024
try :
    serverPort = int(os.getenv('serverPort_env'))
    serverIP = os.getenv('serverIP_env')
except TypeError :
    sys.exit('\033cPlease open .env.shared and follow instructions')
serverAddress = (serverIP,serverPort)

UDPClient = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

def make_prediction(Dataset) -> int :
    Loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in Loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
            predicted = torch.argmax(model(video_frames, imu_data))
    return predicted.item()

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

    print(f'\n{LINE_UP}', end=LINE_CLEAR)
except KeyboardInterrupt :
    sys.exit('\nProgramme Stopped\n')

Start_Tracking_Time = time.time()

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
idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

model = FusionModel(config.MODEL.MoViNetA0, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

TimeFromClient = str(time.time()-Start_Tracking_Time)
print(TimeFromClient)
TimeFromClient_bytes = TimeFromClient.encode('utf-8')
UDPClient.sendto(TimeFromClient_bytes, serverAddress)


try : # Main Loop
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n\n\n')
    sample_num = ''
    first_sample_num = ''
    Motor_activation_counter = 0
    last_action = 'Down'        # The first action is set to Down so the first thing we can do is grab
    last_motor_action = 'Down'

    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action

    prediction_save = [] # prediction_save[-1] is the newest prediction, and prediction_save[-prediction_threshold] is the oldest saved
    for i in range(prediction_threshold) :
        prediction_save.append('')
    
    while True:
        while sample_num == dataset.SampleNumber : # We check for new sample every millisecond
            time.sleep(0.001)
            try :
                dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
            except IndexError :
                time.sleep(1)
                dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
        sample_num = dataset.SampleNumber
        if first_sample_num == '' : first_sample_num = sample_num # We get the number of the first sample


        try :
            prediction = make_prediction(dataset)
        except FileNotFoundError : 
            print('Folder Got deleted')
            raise KeyboardInterrupt

        tracking[prediction] += 1

        for i in range(prediction_threshold,1,-1) :
            prediction_save[-i] = prediction_save[-i+1]
        prediction_save[-1] = idx_to_action.get(prediction)

        print(LINE_UP, end=LINE_CLEAR)
        print(LINE_UP, end=LINE_CLEAR)

        if all_the_same(prediction_save)[0] :
            Motor_activation_counter += 1

            if prediction_save[-1] == 'grab' :
                if last_action != 'Grab' and last_motor_action != 'Grab':
                    last_action = 'Grab'
                    last_motor_action = 'Grab'
                    print(f'Action {Motor_activation_counter} is {last_action} at {round(time.time()-Start_Tracking_Time,2)}s')

                    messageFromClient = 'Grab'
                    messageFromClient_bytes = messageFromClient.encode('utf-8')
                    UDPClient.sendto(messageFromClient_bytes, serverAddress)

                    print('Waiting For Server Response')
                    Grab_Response, _ = UDPClient.recvfrom(bufferSize)
                    if Grab_Response.decode('utf-8') == 'Grab Received' :
                        print(LINE_UP, end=LINE_CLEAR)
                        print('Grab Done\n')
                    else : print(f'Incorrect Grab Response, message : {Grab_Response.decode('utf-8')}\n\n')
                    
                
                else :
                    print(prediction_save)
                    print ('Grabbing ...')


            elif prediction_save[-1] == 'down' :
                if last_action != 'Down' and last_motor_action != 'Down':
                    last_action = 'Down'
                    last_motor_action = 'Down'
                    print(f'Action {Motor_activation_counter} is {last_action} at {round(time.time()-Start_Tracking_Time,2)}s')

                    messageFromClient = 'Down'
                    messageFromClient_bytes = messageFromClient.encode('utf-8')
                    UDPClient.sendto(messageFromClient_bytes, serverAddress)

                    print('Waiting For Server Response')
                    Down_Response, _ = UDPClient.recvfrom(bufferSize)
                    if Down_Response.decode('utf-8') == 'Down Received' :
                        print(LINE_UP, end=LINE_CLEAR)
                        print('Down Done\n')
                    else : print(f'Incorrect Down Response, message : {Down_Response.decode('utf-8')}\n\n')



                else :
                    print(prediction_save)
                    print('Putting Down ...')



            elif last_action != 'Walk' :
                last_action = 'Walk'
                print(f'Action {Motor_activation_counter} is Walk at {round(time.time()-Start_Tracking_Time,2)}s\n\n')

                messageFromClient = 'Walk'
                messageFromClient_bytes = messageFromClient.encode('utf-8')
                UDPClient.sendto(messageFromClient_bytes, serverAddress)

                print('Waiting For Server Response')
                Walk_Response, _ = UDPClient.recvfrom(bufferSize)
                if Walk_Response.decode('utf-8') == 'Walk Received' :
                    print(LINE_UP, end=LINE_CLEAR)
                    print('Down Done\n')
                else : print(f'Incorrect Down Response, message : {Walk_Response.decode('utf-8')}\n\n')
            
            else : 
                print(prediction_save)
                print('Walking ...')

        
        else :
            print(prediction_save)
            print(f'{sample_num} : {idx_to_action.get(prediction)}')

        

        




        


except KeyboardInterrupt:
    pass


    
num_of_predictions = 0
for i in tracking :
    num_of_predictions += i
num_first = int(first_sample_num.replace('Sample_',''))
num_last = int(sample_num.replace('Sample_',''))

print(f'num_first : {num_first}\nnum_last : {num_last}\nnum of prediction : {num_of_predictions}')

if num_of_predictions > 1 : end_text_prediction = 's'
else : end_text_prediction = ''
print(f'\nThere were a total of {num_of_predictions} prediction{end_text_prediction}, with {(num_last-num_first+1)-num_of_predictions} missed')
for action, i in action_to_idx.items() :
    print(f'{tracking[i]} for {action}')

if STOP_ALL :
    #os.system('pkill -f GetData.py') # Stops GetData.py #Dose not work in windows + dose not stop Get Data cleanly
    messageFromClient = 'Done'
    messageFromClient_bytes = messageFromClient.encode('utf-8')
    UDPClient.sendto(messageFromClient_bytes, serverAddress)

if __name__ == "__main__" :
    print('\nProgramme Stopped\n')
