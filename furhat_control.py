from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from furhat_realtime_api import FurhatClient


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    relative: bool = False


class FurhatController:
    """
    Thin wrapper around Furhat Realtime API (furhat-realtime-api on PyPI).

    Note: The public Realtime API does not expose a direct "get current head pose"
    call. We therefore cache the last commanded head pose and return it on request.
    """

    def __init__(self, ip_address: str):
        self.ip_address = ip_address
        self.client = FurhatClient(ip_address)
        self._last_head_pose: Optional[HeadPose] = None

    def connect(self) -> None:
        self.client.connect()

    def disconnect(self) -> None:
        self.client.disconnect()

    def set_head_pose(
        self,
        yaw: float,
        pitch: float,
        roll: float,
        relative: bool = False,
    ) -> None:
        """
        Change the robot's head pose.

        yaw, pitch, roll are in degrees.
        relative=True makes the pose relative to the current attention target.
        """
        self.client.request_face_headpose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            relative=relative,
        )
        self._last_head_pose = HeadPose(yaw=yaw, pitch=pitch, roll=roll, relative=relative)

    def move_head_relative(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Move head relative to the current position (degrees).
        """
        self.set_head_pose(yaw=yaw, pitch=pitch, roll=roll, relative=True)

    def move_head_absolute(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Move head to an absolute position (degrees).
        """
        self.set_head_pose(yaw=yaw, pitch=pitch, roll=roll, relative=False)

    def get_head_pose(self) -> Optional[HeadPose]:
        """
        Returns the last commanded head pose, or None if never set.
        """
        return self._last_head_pose


def connect_furhat(ip_address: str) -> FurhatController:
    controller = FurhatController(ip_address)
    controller.connect()
    return controller


def change_head_movement(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
    relative: bool = False,
) -> None:
    controller.set_head_pose(yaw=yaw, pitch=pitch, roll=roll, relative=relative)


def move_head_relative(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
) -> None:
    controller.move_head_relative(yaw=yaw, pitch=pitch, roll=roll)


def move_head_absolute(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
) -> None:
    controller.move_head_absolute(yaw=yaw, pitch=pitch, roll=roll)


def get_current_head_position(
    controller: FurhatController,
) -> Optional[Tuple[float, float, float]]:
    pose = controller.get_head_pose()
    if pose is None:
        return None
    return (pose.yaw, pose.pitch, pose.roll)
