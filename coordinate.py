import numpy
from numpy import sin,cos,pi

# ---------- arm length ----------
# do it later when have data

dh_parameter = [{'alpha':0, 'r':20, 'd':0},
                {'alpha':0, 'r':13, 'd':0},
                {'alpha':0, 'r':10, 'd':0}                
        ]


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
                               [sin_theta,  cos_theta*sin_alpha, -cos_theta*sin_alpha, dh['r']*sin_theta],
                               [0,                    sin_alpha,            cos_alpha,           dh['d']],
                               [0,0,0,1]
                        ])

    return tranformation

def inverse_matrix(x:float, y:float)  :
    # include a link length
    link1 = dh_parameter[0]['r']
    link2 = dh_parameter[1]['r']
    if (x**2+y**2) > (link1**2+link2**2) :
        raise ValueError("your coordinate destination is out of range")
    
    
    r = numpy.sqrt((x**2)+(y**2))
    phi_1 = numpy.acos((link2**2-(link1**2+r**2))/2*link1*r)
    phi_2 = numpy.atan(y/x)
    theta_1 = phi_2-phi_1

    phi_3 = numpy.acos((r**2-(link1**2+link2**2))/2*link1*link2)
    theta_2 = (pi/2)-phi_3
    
    return theta_1, theta_2

