import pygame
import socket
import threading
import time
import os

# Screen dimensions (common for 3.5" RPi screens)
# XPT2046 screens are often 480x320 or 320x240
WIDTH, HEIGHT = 480, 320
# For XPT2046, you might need to specify the framebuffer
# os.environ["SDL_FBDEV"] = "/dev/fb1"

# Colors
BLACK = (0, 0, 0)
CYAN = (0, 255, 255)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

class RobotFace:
    def __init__(self):
        pygame.init()
        # Set up display
        # If running on RPi with FB, use: pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Robot Face")
        self.clock = pygame.time.Clock()
        self.running = True
        self.emotion = "neutral"
        self.blink_timer = 0
        self.is_blinking = False

    def draw_eye(self, x, y, size, emotion):
        if self.is_blinking:
            # Drawing a thin line for blink
            pygame.draw.line(self.screen, CYAN, (x - size, y), (x + size, y), 5)
            return

        color = CYAN
        if emotion == "angry":
            color = RED
            # Angled eyes
            points = [(x - size, y - size), (x + size, y), (x - size, y + size)]
            pygame.draw.polygon(self.screen, color, points)
        elif emotion == "sad":
            color = CYAN
            # Droopy eyes
            pygame.draw.circle(self.screen, color, (x, y), size)
            pygame.draw.rect(self.screen, BLACK, (x - size, y - size, size * 2, size))
        elif emotion == "happy":
            color = CYAN
            # Curved eyes (arc)
            pygame.draw.arc(self.screen, color, (x - size, y - size, size * 2, size * 2), 0, 3.14, 10)
        elif emotion == "surprised":
            color = YELLOW
            pygame.draw.circle(self.screen, color, (x, y), size + 10, 5)
            pygame.draw.circle(self.screen, color, (x, y), size // 2)
        else: # neutral
            pygame.draw.circle(self.screen, color, (x, y), size)
            # Subtle inner glow
            pygame.draw.circle(self.screen, WHITE, (x - size//3, y - size//3), size//4)

    def draw(self):
        self.screen.fill(BLACK)
        eye_size = 60
        spacing = 100
        self.draw_eye(WIDTH // 2 - spacing, HEIGHT // 2, eye_size, self.emotion)
        self.draw_eye(WIDTH // 2 + spacing, HEIGHT // 2, eye_size, self.emotion)
        pygame.display.flip()

    def update_emotion(self, new_emotion):
        print(f"Switching to emotion: {new_emotion}")
        self.emotion = new_emotion

    def run_socket_server(self):
        # UDP server to receive emotion commands
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", 5005))
        print("Face Server listening on UDP port 5005...")
        while self.running:
            data, addr = sock.recvfrom(1024)
            cmd = data.decode("utf-8").strip().lower()
            if cmd in ["happy", "sad", "angry", "neutral", "surprised", "blink"]:
                self.update_emotion(cmd)

    def main_loop(self):
        # Start socket thread
        threading.Thread(target=self.run_socket_server, daemon=True).start()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            # Random blinking logic
            self.blink_timer += 1
            if not self.is_blinking and self.blink_timer > 100:
                if time.time() % 3 < 0.1: # randomish
                    self.is_blinking = True
                    self.blink_timer = 0
            if self.is_blinking and self.blink_timer > 5:
                self.is_blinking = False
                self.blink_timer = 0

            self.draw()
            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    face = RobotFace()
    face.main_loop()
