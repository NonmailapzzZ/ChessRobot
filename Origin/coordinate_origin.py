import numpy as np
from numpy import sin,cos,pi

# ---------- arm length ----------
# do it later when have data

dh_parameter = [{'alpha':0, 'r':24.833, 'd':0},
                {'alpha':pi, 'r':22.338, 'd':0},
                {'alpha':0, 'r':0, 'd':0}                
        ]


def camera_plane2robot_plane(x_coordiante, y_coordinate) :
    rotation_matrix = []
    translation = []
    cam_location = np.array([x_coordiante, y_coordinate], np.float16)
    cam_location.transpose()

    x = cam_location[0]
    y = cam_location[1]

    return x, y
    

# Tranfomation matrix for each
def tranformation_matrix(theta:int, index:int) -> np.array :
    
    if index >= len(dh_parameter):
        raise ValueError(f"Joint index {index} is out of range.")
    
    dh = dh_parameter[index]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(dh['alpha'])
    sin_alpha = np.sin(dh['alpha'])
    
    tranformation= np.array([[cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, dh['r']*cos_theta],
                               [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, dh['r']*sin_theta],
                               [0,                    sin_alpha,            cos_alpha,           dh['d']],
                               [0,0,0,1]
                        ])

    # return (tranfomation matrix, x-coordination, y-coordination)
    return (tranformation, tranformation[0,3], tranformation[1,3]) 

def inverse_matrix(x:np.float16, y:np.float16) :
    #- - - const. - - -
    DANGER_MIN = -180
    DANGER_MAX = 0
    #- - - - - - - - - -
    
    
    x, y =  camera_plane2robot_plane(x, y)
    # x = numpy.float64(x)
    # y = numpy.float64(y)
    # include a link length
    link1 = np.float64(dh_parameter[0]['r'])
    link2 = np.float64(dh_parameter[1]['r'])
    if (np.sqrt(x**2+y**2) > abs(link1+link2)) or (np.sqrt(x**2+y**2) < abs(link1-link2)):
        lenght = np.sqrt(x**2+y**2)
        raise ValueError(f"your coordinate destination is out of range ,its {lenght}, arm can be",(link1+link2))
    
    
    r = np.sqrt((x**2)+(y**2))
    phi_1 = np.arccos((link2**2 - link1**2 - r**2)/(-2*link1*r))
    phi_2 = np.arctan(y/x)
   
    theta_1_down = phi_2 - phi_1
    phi_3 = np.arccos((r**2 - link1**2 - link2**2)/(-2*link1*link2))
    theta_2_down = pi-phi_3
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elbow_is_danger = (theta_1_down > DANGER_MIN) and (theta_1_down < DANGER_MAX) 
    
    if elbow_is_danger:
        # recalculate on theta_1 and theta_2
        phi_1_up = -phi_1 
        theta_1_up = phi_2 - phi_1_up
        theta_2_up = -(theta_2_down)
        # for send theta_1 to 300deg servo need to invert because calculation on Counter-clockwise
                                        # but for control use clockwise
        return np.rad2deg(theta_1_up) , np.rad2deg(theta_2_up)
    else :
        return np.rad2deg(theta_1_down), np.rad2deg(theta_2_down)
    
    


# print(inverse_matrix(5,10))