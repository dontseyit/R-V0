
from base_ctrl import BaseController
import pyttsx3
from typing import Optional


class Command:
    """Send drive commands and optionally issue speech prompts."""

    def __init__(self, uart_dev: str = "/dev/ttyAMA0", baud_rate: int = 115200) -> None:
        self.base = BaseController(uart_dev, baud_rate)
        self._engine: Optional[pyttsx3.Engine] = None

    def _ensure_engine(self) -> pyttsx3.Engine:
        if self._engine is None:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 180)
        return self._engine

    def control_speed(self, input_left, input_right):
        """ Control the speed of the left and right motors. 
        
        send_command({"T":1,"L":input_left,"R":input_right})
        """
        self.base.base_speed_ctrl(input_left, input_right)

    def drive(self, left_pwm: float, right_pwm: float) -> None:
        self.base.send_command({"T": 11, "L": int(left_pwm), "R": int(right_pwm)})

    def play_speech(self, input_text: Optional[str]) -> None:
        if not input_text:
            return
        engine = self._ensure_engine()
        engine.say(input_text)
        engine.runAndWait()

    def move_camera(self, input_x: int, input_y: int, input_speed: int = 20, input_acc: int = 0) -> None:
        """Controls the camera's movement.
        
        send_command({"T":133,"X":input_x,"Y":input_y,"SPD":input_speed,"ACC":input_acc})
        """
        self.base.gimbal_ctrl(input_x, input_y, input_speed, input_acc)