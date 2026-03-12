import math
from typing import Dict

class Mapper():
    """
    Does the mapping calculation according to the camera dimensions
    Assumes that camera sits directly on top of the center of the robot

    """
    def __init__(self, height : int, width : int, fov_x : int, fov_y : int, cam_to_robot : float ):
        """
        Inializaion

        Args:
            height: pixel height in camera stream
            width: pixel width in camera stream
            fov_x: field of view in x direction of camera in degrees
            fov_y: field of view in y direction of camera in degrees
            cam_to_robot: height between camera and robot in meters
        """
        self.width = width
        self.height = height
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.cam_to_robot = cam_to_robot

        ## calculate center of frame, especially adjust for height difference of robot and camera
        self.width_center = self.width / 2 
        self.height_center = self.height / 2

        self.focal_legnth_x = width / (2 * math.tan(self.fov_x / 2))
        self.focal_legnth_y = width / (2* math.tan(self.fov_y / 2))

        ## assume the robot like 1.2 m off the ground
        self.robot_height = 1.2
        self.cam_height = self.cam_to_robot + self.robot_height


    def get_yaw(self, x: int, y : int) -> float:
        """
        
        Maps the clicked location (x, y) on screen to left and right (yaw) angle for the robot.
        formula to convert pixels to angle is: 
        arctan((x - c_x) / f_x)
        where f_x is focal length, estimated using:
        (image width) / (2 tan(field of view in x / 2))

        Args:
            x: integer x coordiante of pixel
            y: integer y coordiante of pixel 

        Return:
            yaw_angle: angle of yaw in radians
        """

        yaw_angle = math.atan((x - self.width / 2) / self.focal_legnth_x)

        return yaw_angle


    def get_pitch(self, x : int, y : int) -> float:
        """
        Maps the clicked location (x, y) on screen to up and down (pitch) angle for the robot.
        formula to convert pixels to angle is: 
        arctan((y - c_y) / f_y)
        where f_y is focal length, estimated using:
        (image width) / (2 tan(field of view in y / 2))

        Args:
            x: integer x coordiante of pixel
            y: integer y coordiante of pixel 

        Return:
            robot_pitch: angle of pitch in radians
        """
        ## pitch angle relative to camera's axis
        pitch_camera = math.atan((y - self.height / 2) / self.focal_legnth_y)
        distance_to_target = self.cam_height / math.tan(pitch_camera)
        robot_pitch = math.atan(self.robot_height / distance_to_target)
        return robot_pitch
    

    def get_absolute_movement(self, x : int, y :int , bbox : Dict[str, int]) -> tuple[float, float, float]:
        """
        Get the absolute amount to move in roobt's plane. Relative to robot at 0,0,0

        Args:
            x: integer x coordiante of pixel
            y: integer y coordiante of pixel 
            bbox: bbox of face

        Return:
            pitch in radians
            yaw in radians
            roll in radians
        """

        # first get yaw
        raw_yaw = self.get_yaw(x, y)
        raw_pitch = self.get_pitch(x, y)
        
        # get bbox proportion, scale down movements by scale
        proportion = self.scale(bbox) 

        return raw_pitch * proportion, raw_yaw * proportion, 0
    
    def get_relative_movement(self, x : int, y :int , bbox : Dict[str, int],
                              cur_pitch : int = 0, cur_yaw : int = 0, cur_roll : int = 0) -> tuple[float, float, float]:
        
        """
        Get the relative amount to move in roobt's plane. Relative to robot at pitch, yaw, roll

        Args:
            x: integer x coordiante of pixel
            y: integer y coordiante of pixel 
            bbox: bbox of face
            cur_pitch: current pitch of robot in radians
            cur_yaw: current yaw of robot in radians
            cur_roll: current roll of robot in radians

        Return:
            pitch in radians
            yaw in radians
            roll in radians
        """
        raw_pitch, raw_yaw, raw_roll = self.get_absolute_movement(x, y, bbox)

        if raw_pitch > cur_pitch:
            pitch = raw_pitch - cur_pitch
        else:
            pitch = cur_pitch - raw_pitch

        if raw_yaw > cur_yaw:
            yaw = raw_yaw - cur_yaw
        else:
            yaw = cur_yaw - raw_yaw

        return pitch, yaw, 0


    def scale(self, bbox : Dict[str, int]):
        ## estimates distance & proportion to scale stuff scale by 1 - proportion

        ## proportion to screen size
        x1 = bbox['x1']
        x2 = bbox['x2']
        y1 = bbox['y1']
        y2 = bbox['y2']

        proportion = self.height * self.width / (abs(x1 - x2) * abs(y1 - y2))

        # scalling down proportion
        return 1 - proportion * 0.8









