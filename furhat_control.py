from __future__ import annotations

from dataclasses import dataclass
import math
import time
import asyncio
import threading
from typing import Optional, Tuple

from furhat_realtime_api import AsyncFurhatClient


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
        self.client = AsyncFurhatClient(ip_address)
        self._last_head_pose: Optional[HeadPose] = None
        self._last_move_time: Optional[float] = None
        self._min_move_interval_s = 0.5
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="furhat-async-loop", daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro: asyncio.Future) -> asyncio.Future:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def connect(self) -> None:
        await self.client.connect()

    async def disconnect(self) -> None:
        await self.client.disconnect()

    async def set_head_pose(
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
        if not self.can_move_now():
            remaining = self.time_until_move()
            raise RuntimeError(
                f"Head movement rate-limited. Try again in {remaining:.3f}s."
            )


        if relative:
            await self.client.send_event({"type":"request.face.headpose",
                                            "yaw":yaw, "pitch":pitch, "roll":roll,
                                            "relative" : "true"})

        else:

            await self.client.send_event({"type":"request.face.headpose",
                                            "yaw":yaw, "pitch":pitch, "roll":roll,
                                            "relative" : "false"})

        await asyncio.sleep(0.1)
        self._last_move_time = time.monotonic()
        self._last_head_pose = HeadPose(yaw=yaw, pitch=pitch, roll=roll, relative=relative)
        print(
            f"Furhat head target: pitch={pitch:.3f}, yaw={yaw:.3f}, roll={roll:.3f}, "
            f"relative={relative}"
        )

    async def move_head_relative(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Move head relative to the current position (input in radians).
        """
        #asyncio.run(self.set_head_pose(
        #    yaw=math.degrees(yaw),
        #    pitch=math.degrees(pitch),
        #    roll=math.degrees(roll),
        #    relative=True,
        #))

        await self.set_head_pose(
            yaw=math.degrees(yaw),
            pitch=math.degrees(pitch),
            roll=math.degrees(roll),
            relative=True,
        )

    async def move_head_absolute(self, yaw: float, pitch: float, roll: float) -> None:
        """
        Move head to an absolute position (input in radians).
        """
        await (self.set_head_pose(
            yaw=math.degrees(yaw),
            pitch=math.degrees(pitch),
            roll=math.degrees(roll),
            relative=False,
        ))

    def get_head_pose(self) -> Optional[HeadPose]:
        """
        Returns the last commanded head pose, or None if never set.
        """
        return self._last_head_pose

    def can_move_now(self) -> bool:
        if self._last_move_time is None:
            return True
        return (time.monotonic() - self._last_move_time) >= self._min_move_interval_s

    def time_until_move(self) -> float:
        if self._last_move_time is None:
            return 0.0
        remaining = self._min_move_interval_s - (time.monotonic() - self._last_move_time)
        return max(0.0, remaining)

    def get_head_pose_from_robot(self) -> Optional[HeadPose]:
        """
        Attempt to fetch the current head pose from the robot.

        Note: The furhat-realtime-api (per PyPI docs) does not expose a direct
        "get head pose" request. This method currently cannot query live pose
        and will raise to make the limitation explicit.
        """
        raise NotImplementedError(
            "furhat-realtime-api does not provide a head pose getter; "
            "use get_head_pose() for the last commanded pose instead."
        )


def _run_async(controller: FurhatController, coro):
    return controller.submit(coro).result()


def connect_furhat(ip_address: str) -> FurhatController:
    controller = FurhatController(ip_address)
    _run_async(controller, controller.connect())
    return controller


def disconnect_furhat(controller: FurhatController) -> None:
    _run_async(controller, controller.disconnect())


def change_head_movement(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
    relative: bool = False,
) -> None:
    _run_async(controller, controller.set_head_pose(yaw=yaw, pitch=pitch, roll=roll, relative=relative))


def move_head_relative(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
) -> None:
    _run_async(controller, controller.move_head_relative(yaw=yaw, pitch=pitch, roll=roll))


def move_head_absolute(
    controller: FurhatController,
    yaw: float,
    pitch: float,
    roll: float,
) -> None:
    _run_async(controller, controller.move_head_absolute(yaw=yaw, pitch=pitch, roll=roll))


def get_current_head_position(
    controller: FurhatController,
) -> Optional[Tuple[float, float, float]]:
    pose = controller.get_head_pose()
    if pose is None:
        return None
    return (pose.yaw, pose.pitch, pose.roll)
