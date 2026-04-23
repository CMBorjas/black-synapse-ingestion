import socket
import logging
import os

logger = logging.getLogger(__name__)

class FaceController:
    """
    Sends emotion commands to the Raspberry Pi face display via UDP over Ethernet.
    """
    def __init__(self):
        self.rpi_ip = os.getenv("RPI_FACE_IP", "192.168.1.100")
        self.rpi_port = int(os.getenv("RPI_FACE_PORT", "5005"))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def set_emotion(self, emotion: str):
        """
        Send an emotion command to the RPi.
        Supported: happy, sad, angry, neutral, surprised
        """
        try:
            logger.info(f"Sending emotion '{emotion}' to RPi at {self.rpi_ip}:{self.rpi_port}")
            message = emotion.encode("utf-8")
            self.sock.sendto(message, (self.rpi_ip, self.rpi_port))
        except Exception as e:
            logger.error(f"Failed to send emotion to RPi: {e}")

# Global instance
face_controller = FaceController()
