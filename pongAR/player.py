import cv2
import numpy as np

class Player:
    pass


if __name__ == '__main__':
    from ball import Ball
    from random import randint
    cnv = np.ones((600,600,3), dtype=np.uint8() )
    
        
    mrB = Ball(300+ randint(0,100),
               300+ randint(0,100), 50, (0,255,255))

    
    while True:
        mrB.move(cnv)
        
        cv2.imshow('main', cnv)

        key = cv2.waitKey(1)

        if key == 27:
            break
    cv2.destroyAllWindows()
