import os
import sys
import time
import signal
import cv2
import numpy
from robobo_interface.base import IRobobo
from robobo_interface.datatypes import (
    Emotion,
    LedColor,
    LedId,
    Acceleration,
    Position,
    Orientation,
    WheelPosition,
    SoundEmotion,
)
from robobo_interface.utils import LockedSet
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from typing import Callable, List, NoReturn, Optional, TypeVar
from numpy.typing import NDArray

T = TypeVar("T")


class SimulationRobobo(IRobobo):
    def __init__(
        self,
        identifier: int = 0,
        api_port: Optional[int] = None,
        ip_adress: Optional[str] = None,
        logger: Callable[[str], None] = print,
        timeout_dur: int = 10,
    ):
        self._logger = logger
        self._used_pids: LockedSet[int] = LockedSet()
        self._identifier = f"[{identifier}]"

        if api_port is None:
            api_port = int(os.getenv("COPPELIA_SIM_PORT", "23000"))

        if ip_adress is None:
            ip_adress = os.getenv("COPPELIA_SIM_IP", "0.0.0.0")

        try:
            self._client = timeout(
                lambda: RemoteAPIClient(host=ip_adress, port=api_port), timeout_dur
            )
        except TimeoutError:
            self._fail_connect(api_port, ip_adress)

        try:
            self._sim = timeout(lambda: self._client.require("sim"), timeout_dur)
        except TimeoutError:
            self._fail_connect(api_port, ip_adress)

        self._initialise_handles()
        self._logger(
            f"""Connected to remote CoppeliaSim API server at port {api_port}
            Connected to robot: {self._identifier}"""
        )
        self.bl = 0 

    def set_emotion(self, emotion: Emotion) -> None:
        self._logger(f"The robot shows {emotion.value} on its screen")

    def move(
        self,
        left_speed: int,
        right_speed: int,
        millis: int,
        blockid: Optional[int] = None,
    ) -> int:
        if not self.is_running():
            raise RuntimeError("Cannot move wheels when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)
        # print(f"Added blockid {blockid} to _used_pids: {self._used_pids}")

        self._sim.callScriptFunction(
            "moveWheelsByTime",
            self._wheels_script,
            [right_speed, left_speed],
            [millis / 1000.0],
            [self._block_string(blockid)],
            bytearray(),
        )

        try:
            self.block_until_free(blockid)
        except Exception as e:
            # print(f"Error in block_until_free: {e}")
            self._used_pids.discard(blockid)
            raise

        return blockid

    def block_until_free(self, blockid: int) -> None:
        while self.is_blocked(blockid):
            time.sleep(0.000002)
        # print(f"Blockid {blockid} is now free. Current used PIDs: {self._used_pids}")
        self._used_pids.discard(blockid)

    def reset_wheels(self) -> None:
        if not self.is_running():
            raise RuntimeError("Cannot reset wheels when simulation is not running")
        self._sim.callScriptFunction(
            "resetWheelEncoders",
            self._wheels_script,
            [],
            [],
            [],
            bytearray(),
        )

    def talk(self, message: str) -> None:
        self._logger(f"The robot {self._identifier} says: {message}")

    def play_emotion_sound(self, emotion: SoundEmotion) -> None:
        self._logger(f"The robot {self._identifier} makes sound: {emotion.value}")

    def set_led(self, selector: LedId, color: LedColor) -> None:
        if not self.is_running():
            raise RuntimeError("Cannot set leds when simulation is not running")
        self._sim.callScriptFunction(
            "setLEDColor",
            self._leds_script,
            [],
            [],
            [selector.value, color.value],
            bytearray(),
        )

    def read_irs(self) -> List[Optional[float]]:
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readAllIRSensor",
            self._ir_script,
            [],
            [],
            [],
            bytearray(),
        )
        return list(ints)

    def get_image_front(self) -> NDArray[numpy.uint8]:
        img, [resX, resY] = self._sim.getVisionSensorImg(self._smartphone_camera)
        img = numpy.frombuffer(img, dtype=numpy.uint8).reshape(resY, resX, 3)
        img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
        return img

    def set_phone_pan(
        self, pan_position: int, pan_speed: int, blockid: Optional[int] = None
    ) -> int:
        if not self.is_running():
            raise RuntimeError("Cannot set phone pan when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)

        self._sim.callScriptFunction(
            "movePanTo",
            self._pan_motor_script,
            [pan_position, pan_speed],
            [],
            [self._block_string(blockid)],
            bytearray(),
        )

        return blockid

    def read_phone_pan(self) -> int:
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readPanPosition",
            self._pan_motor_script,
            [],
            [],
            [],
            bytearray(),
        )
        return int(ints[0])

    def set_phone_tilt(
        self, tilt_position: int, tilt_speed: int, blockid: Optional[int] = None
    ) -> int:
        if not self.is_running():
            raise RuntimeError("Cannot set phone tilt when simulation is not running")
        if blockid in self._used_pids:
            raise ValueError(f"BlockID {blockid} is already in use: {self._used_pids}")
        blockid = blockid if blockid is not None else self._first_unblocked()
        self._used_pids.add(blockid)

        self._sim.callScriptFunction(
            "moveTiltTo",
            self._tilt_motor_script,
            [tilt_position, tilt_speed],
            [],
            [self._block_string(blockid)],
            bytearray(),
        )

        return blockid

    def read_phone_tilt(self) -> int:
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readTiltPosition",
            self._tilt_motor_script,
            [],
            [],
            [],
            bytearray(),
        )
        return int(ints[0])

    def read_accel(self) -> Acceleration:
        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "readAccelerationSensor",
            self._smartphone_script,
            [],
            [],
            [],
            bytearray(),
        )
        return Acceleration(*floats)

    def read_orientation(self) -> Orientation:
        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "readOrientationSensor",
            self._smartphone_script,
            [],
            [],
            [],
            bytearray(),
        )
        return Orientation(*floats)

    def read_wheels(self) -> WheelPosition:
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "readWheels",
            self._wheels_script,
            [],
            [],
            [],
            bytearray(),
        )
        return WheelPosition(*ints)

    def sleep(self, seconds: float) -> None:
        start_time = self.get_sim_time()
        while self.get_sim_time() - start_time < seconds:
            if not self.is_running():
                raise RuntimeError("Cannot sleep when simulation is not running")
            time.sleep(0.02)

    def is_blocked(self, blockid: int) -> bool:
        return False
        # res = self._sim.getInt32Signal(self._block_string(blockid))
        
        # if res == 0:
        #     self._used_pids.discard(blockid)
        #     # print(f"Released blockid {blockid}. Current used PIDs: {self._used_pids}")
        #     return False
        # else:
        #     self._used_pids.add(blockid)
        #     return False
            

    def block(self) -> None:
        while any(self.is_blocked(blockid) for blockid in self._used_pids):
            time.sleep(0.0000000000002)

    def play_simulation(self) -> None:
        self._sim.startSimulation()

    def pause_simulation(self) -> None:
        self._sim.pauseSimulation()
        while not self.is_paused():
            time.sleep(0.002)

    def stop_simulation(self) -> None:
        self._sim.stopSimulation()
        while not self.is_stopped():
            time.sleep(0.002)

    def is_stopped(self) -> bool:
        return self._sim.getSimulationState() == self._sim.simulation_stopped

    def is_paused(self) -> bool:
        return self._sim.getSimulationState() == self._sim.simulation_paused

    def is_running(self) -> bool:
        state = self._sim.getSimulationState()
        return (state != self._sim.simulation_stopped) and (state != self._sim.simulation_paused)

    def get_sim_time(self) -> float:
        return self._sim.getSimulationTime()

    def nr_food_collected(self) -> int:
        ints, _floats, _strings, _buffer = self._sim.callScriptFunction(
            "remote_get_collected_food",
            self._food_script,
            [],
            [],
            [],
            bytearray(),
        )
        return ints[0]

    def get_position(self) -> Position:
        pos = self._sim.getObjectPosition(self._robobo, self._sim.handle_world)
        return Position(*pos)

    def get_orientation(self) -> Orientation:
        orient = self._sim.getObjectOrientation(self._robobo, self._sim.handle_world)
        return Orientation(*orient)

    def set_position(self, position: Position, orientation: Orientation) -> None:
        self._sim.setObjectPosition(
            self._robobo, self._sim.handle_world, [position.x, position.y, position.z]
        )
        self._sim.setObjectOrientation(
            self._robobo,
            self._sim.handle_world,
            [orientation.yaw, orientation.pitch, orientation.roll],
        )

    def base_position(self) -> Position:
        if self._base is None:
            raise AttributeError("Scene does not have a base")
        pos = self._sim.getObjectPosition(self._base, self._sim.handle_world)
        return Position(*pos)

    def base_detects_food(self) -> bool:
        return self._base_food_distance() > 0

    def _base_food_distance(self) -> float:
        if self._base is None:
            raise AttributeError("Scene does not have a base")
        if self._food_script is None:
            raise AttributeError("Cannot find any food in the scene")
        _ints, floats, _strings, _buffer = self._sim.callScriptFunction(
            "getFoodDistance",
            self._base_script,
            [],
            [],
            [],
            bytearray(),
        )
        ret = floats[0]
        if ret < 0:
            raise AttributeError("Cannot find any food in the scene")
        return ret

    def _block_string(self, blockid: int) -> str:
        return f"Block_{self._identifier}_{blockid}"

    def _initialise_handles(self) -> None:
        self._robobo = self._get_object(f"/Robobo{self._identifier}")
        self._wheels_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/Left_Motor"))
        self._leds_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/Back_L"))
        self._ir_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/IR_Back_C"))
        self._pan_motor_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/Pan_Motor"))
        self._tilt_motor_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor"))
        self._smartphone_script = self._get_childscript(self._get_object(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor/Smartphone_Respondable"))
        self._smartphone_camera = self._get_object(f"/Robobo{self._identifier}/Pan_Motor/Pan_Respondable/Tilt_Motor/Smartphone_Respondable/Smartphone_camera")

        try:
            self._base = self._get_object("/Base")
            self._base_script = self._get_childscript(self._base)
        except AttributeError:
            self._base = None
            self._base_script = None

        try:
            self._food_script = self._get_childscript(self._get_object("/Food"))
        except AttributeError:
            self._food_script = None

    def _get_object(self, name: str) -> int:
        try:
            ret = self._sim.getObject(name)
        except:
            raise AttributeError(f"Could not find {name} in scene")
        if ret < 0:
            raise AttributeError(f"Could not find {name} in scene")
        return ret

    def _get_childscript(self, obj_handle: int) -> int:
        try:
            ret = self._sim.getScript(self._sim.scripttype_childscript, obj_handle)
        except:
            raise AttributeError(f"Could not find Script of {obj_handle} in scene")
        if ret < 0:
            raise AttributeError(f"Could not find Script of {obj_handle} in scene")
        return ret

    def _fail_connect(self, api_port: int, ip_adress: str) -> NoReturn:
        self._logger(
            """CoppeliaSim Api Connection Error
            Failed connecting to remote API server
            Is the simulation running / playing?

            If not on Linux with --net=host:
            Did you specify the IP adress of your computer in scripts/setup.bash?
            """
        )
        self._logger(f"Looked for API at port: {api_port} at IP adress: {ip_adress}")
        quit_hard()


def timeout(func: Callable[[], T], timeout_duration: int = 10) -> T:
    def handler(_signum, _frame):
        raise TimeoutError()
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        return func()
    finally:
        signal.alarm(0)


def quit_hard() -> NoReturn:
    os.kill(os.getpid(), signal.SIGKILL)
    sys.exit(1)
