############## IMPORT ##############
import torch
import torch.nn as nn
################ DEFINE PARAMETERS AND HYPERPARAMETERS #################
use_qtable = False
NUM_EPISODES = 1000
NUM_EPISODES_TEST = 100
MAX_EPSILON = 1 # Initial exploration probability
MIN_EPSILON = 0.01 # Final exploration probability
GAMMA = 0.99 # discount factor q table
ALPHA = 0.001 # learning rate q table
GAMMA_NN =0.999 #discount factor for neural network
LR = 0.0005 # learning rate neural network
BUFFER_SIZE = 100000
BATCH_SIZE = 64
EPSILON_DECAY = 0.99
TARGET_FREQ_UPDATE = 10
BATCH_SIZE = 64

DEVICE ="cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT = "checkpoints/checkpointDouble1000.pth.tar"


######################## disccretize the continous space to build the Q table##################################
### return the number of the state of the Q_table
def discretize(observation):
    x, y, v_x, v_y, angle, v_angle, left_leg, right_leg = observation
    
    ## discretize the x cordinate into 6 interval
    if x<=-1:
        box = 0
    elif x<=-0.5:
        box = 1
    elif x<=0:
        box = 2
    elif x<=0.5:
        box = 3
    elif x<=1:
        box = 4
    else:
        box = 5
     
    ## discretize the y cordinate into 6 interval
    if y<=-1:
        box += 0
    elif y<=-0.5:
        box += 6
    elif y<=0:
        box += 12
    elif y<=0.5:
        box += 18
    elif y<=1:
        box += 24
    else:
        box += 30
        
    ## discretize the velocity along x axe into 4 interval
    if v_x<=-2.5:
        box += 0
    elif v_x<=-0:
        box += 36
    elif v_x<=2.5:
        box += 72
    else:
        box += 108
        
    ## discretize the velocity along y axe into 4 interval
    if v_y<=-2.5:
        box += 0
    elif v_y<=-0:
        box += 144
    elif v_y<=2.5:
        box += 288
    else:
        box += 432
        
    ## discretize the angle value into 3 interval
    if angle<=-1:
        box += 0
    elif angle<=1:
        box += 576
    else:
        box += 1152

    
    ## discretize the angular velocity value into 4 interval
    if v_angle<=-2.5:
        box += 0
    elif v_angle<=0:
        box += 1728
    elif v_angle<=2.5:
        box += 3456
    else:
        box += 5184
        
    ## boolean left leg touch the ground
    if left_leg==0:
        box += 0
    else:
        box +=6912
        
    ## boolean rigth leg touch the ground
    if right_leg==0:
        box += 0
    else:
        box += 13824
        
    return box  # the maximum state is 27648
############################################################################

#################### NEW DISCRETIZATION ####################################
############# try to increment the number of discrete state to cope with the large precision of the box spaces( float32 con 7 cifre dopo la virgola)
def big_discretize(observation):
    x, y, v_x, v_y, angle, v_angle, left_leg, right_leg = observation
    
    ## discretize the x cordinate into 30 interval
    if x<=-1.4:
        box = 0
    elif x<=-1.3:
        box = 1
    elif x<=-1.2:
        box = 2
    elif x<=-1.1:
        box = 3
    elif x<=-1.0:
        box = 4
    elif x<=-0.9:
        box = 5
    elif x<=-0.8:
        box = 6
    elif x<=-0.7:
        box = 7
    elif x<=-0.6:
        box = 8
    elif x<=-0.5:
        box = 9
    elif x<=-0.4:
        box=10
    elif x<=-0.3:
        box=11
    elif x<=-0.2:
        box = 12
    elif x<=-0.1:
        box = 13
    elif x<=-0.0:
        box = 14
    elif x<=0.1:
        box = 15
    elif x<=0.2:
        box = 16
    elif x<=0.3:
        box = 17
    elif x<=0.4:
        box = 18
    elif x<=0.5:
        box = 19
    elif x<=0.6:
        box = 20
    elif x<=0.7:
        box = 21
    elif x<=0.8:
        box = 22
    elif x<=0.9:
        box=23
    elif x<=1.0:
        box=24
    elif x<=1.1:
        box=25
    elif x<=1.2:
        box=26
    elif x<=1.3:
        box=27
    elif x<=1.4:
        box=28
    else:
        box = 29
    
     
    ## discretize the y cordinate into 30 interval    
    if y<=-1.4:
        box += 0
    elif y<=-1.3:
        box += 30
    elif y<=-1.2:
        box = 60
    elif y<=-1.1:
        box += 90
    elif y<=-1.0:
        box += 120
    elif y<=-0.9:
        box += 150
    elif y<=-0.8:
        box += 180
    elif y<=-0.7:
        box += 210
    elif y<=-0.6:
        box += 240
    elif y<=-0.5:
        box += 270
    elif y<=-0.4:
        box+=300
    elif y<=-0.3:
        box+=330
    elif y<=-0.2:
        box +=360
    elif y<=-0.1:
        box += 390
    elif y<=-0.0:
        box += 420
    elif y<=0.1:
        box += 450
    elif y<=0.2:
        box += 480
    elif y<=0.3:
        box += 510
    elif y<=0.4:
        box += 540
    elif y<=0.5:
        box += 570
    elif y<=0.6:
        box += 600
    elif y<=0.7:
        box += 630
    elif y<=0.8:
        box += 660
    elif y<=0.9:
        box+=690
    elif y<=1.0:
        box+=720
    elif y<=1.1:
        box+=750
    elif y<=1.2:
        box+=780
    elif y<=1.3:
        box+=810
    elif y<=1.4:
        box+=840
    else:
        box += 870
    
        
    ## discretize the velocity along x axe into 21 interval
    if v_x<=-4.5:
        box += 0
    elif v_x<=-4.0:
        box += 900
    elif v_x<=-3.5:
        box += 1800
    elif v_x<=-3.0:
        box += 2700
    elif v_x<=-2.5:
        box += 3600
    elif v_x<=-2.0:
        box += 4500
    elif v_x<=-2.0:
        box += 5400
    elif v_x<=-1.5:
        box += 6300
    elif v_x<=-1.0:
        box += 7200
    elif v_x<=-0.5:
        box += 8100
    elif v_x<= 0.0:
        box+=9000
    elif v_x<= 0.5:
        box+=9900
    elif v_x<= 1.0:
        box += 10800
    elif v_x<= 1.5:
        box =11700
    elif v_x<= 2.0:
        box = 12600
    elif v_x<=2.5:
        box = 13500
    elif v_x<=3.0:
        box = 14400
    elif v_x<=3.5:
        box = 15300
    elif v_x<=4.0:
        box = 16200
    elif v_x<=4.5:
        box = 17100
    else:
        box += 18000
        
        
    ## discretize the velocity along y axe into 21 interval
    if v_y<=-4.5:
        box += 0
    elif v_y<=-4.0:
        box += 18900
    elif v_y<=-3.5:
        box += 37800
    elif v_y<=-3.0:
        box += 56700
    elif v_y<=-2.5:
        box += 75600
    elif v_y<=-2.0:
        box += 94500
    elif v_y<=-2.0:
        box += 113400
    elif v_y<=-1.5:
        box += 132300
    elif v_y<=-1.0:
        box += 151200
    elif v_y<=-0.5:
        box += 170100
    elif v_y<= 0.0:
        box+= 189000
    elif v_y<= 0.5:
        box+= 207900
    elif v_y<= 1.0:
        box += 226800
    elif v_y<= 1.5:
        box +=245700
    elif v_y<= 2.0:
        box += 264600
    elif v_y<=2.5:
        box = 283500
    elif v_y<=3.0:
        box += 302400
    elif v_y<=3.5:
        box +=321300
    elif v_y<=4.0:
        box += 340200
    elif v_y<=4.5:
        box += 369100
    else:
        box += 378000

    ## discretize the angle value into 3 interval
    if angle<=-1:
        box += 0
    elif angle<=1:
        box += 396900
    else:
        box += 793800
        
    ## discretize the angular velocity value into 4 interval
    if v_angle<=-2.5:
        box += 0
    elif v_angle<=0:
        box += 1190700
    elif v_angle<=2.5:
        box += 2381400
    else:
        box += 3572100

    ## boolean left leg touch the ground
    if left_leg==0:
        box += 0
    else:
        box +=4762800
        
    ## boolean rigth leg touch the ground
    if right_leg==0:
        box += 0
    else:
        box += 9525600
        
    return box  #the last state is 19051200(obtained multiplied all the invertval so all the possible combination)


######### FUNCTION FOR SAVE AND LOAD NN #########

def save_model(model, optimizer, episode):  
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CHECKPOINT)

def load_model(file, model, optimizer):
    model_check = torch.load(file, map_location=DEVICE)
    model.load_state_dict(model_check["state_dict"])
    optimizer.load_state_dict(model_check["optimizer"])
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
        
######### END FUNCTION FOR NN ##########
