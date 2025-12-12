import numpy
from numpy import sin,cos,pi

# ---------- arm length ----------
# do it later when have data

dh_parameter = [{'alpha':0, 'r':26.5, 'd':0},
                {'alpha':pi, 'r':26, 'd':0},
                {'alpha':0, 'r':0, 'd':0}                
        ]


def camera_plane2robot_plane(x_coordiante, y_coordinate) :
    rotation_matrix = []
    translation = []
    cam_location = numpy.array([x_coordiante, y_coordinate], numpy.float16)
    cam_location.transpose()

    x = cam_location[0]
    y = cam_location[1]

    return x, y
    

# Tranfomation matrix for each
def tranformation_matrix(theta:int, index:int) -> numpy.array :
    
    if index >= len(dh_parameter):
        raise ValueError(f"Joint index {index} is out of range.")
    
    dh = dh_parameter[index]
    cos_theta = numpy.cos(theta)
    sin_theta = numpy.sin(theta)
    cos_alpha = numpy.cos(dh['alpha'])
    sin_alpha = numpy.sin(dh['alpha'])
    
    tranformation= numpy.array([[cos_theta, -sin_theta*cos_alpha,  sin_theta*sin_alpha, dh['r']*cos_theta],
                               [sin_theta,  cos_theta*cos_alpha, -cos_theta*sin_alpha, dh['r']*sin_theta],
                               [0,                    sin_alpha,            cos_alpha,           dh['d']],
                               [0,0,0,1]
                        ])

    # return (tranfomation matrix, x-coordination, y-coordination)
    return (tranformation, tranformation[0,3], tranformation[1,3]) 

def inverse_matrix(x:numpy.float16, y:numpy.float16) :
    #- - - const. - - -
    DANGER_MIN = -180
    #- - - - - - - - - -
    
    
    x, y =  camera_plane2robot_plane(x, y)
    # x = numpy.float64(x)
    # y = numpy.float64(y)
    # include a link length
    link1 = numpy.float64(dh_parameter[0]['r'])
    link2 = numpy.float64(dh_parameter[1]['r'])
    if (x**2+y**2) > (link1**2+link2**2) :
        raise ValueError("your coordinate destination is out of range")
    
    
    r = numpy.sqrt((x**2)+(y**2))
    phi_1 = numpy.arccos((link2**2 - link1**2 - r**2)/(-2*link1*r))
    phi_2 = numpy.arctan(y/x)
   
    theta_1_down = phi_2 - phi_1
    phi_3 = numpy.arccos((r**2 - link1**2 - link2**2)/(-2*link1*link2))
    theta_2_down = pi-phi_3
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    elbow_is_danger = ()
    
    


# print(inverse_matrix(5,10))