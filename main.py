import numpy
from dataclasses import dataclass
import pygame


@dataclass
class Segment:
    radius: float
    omega: float
    angle: float

    def get_vector(self):
        return self.radius * numpy.exp(2 * numpy.pi * 1j * self.angle)

    def inc_angle(self, dt):
        self.angle += self.omega * 2 * numpy.pi * dt


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    screen_mid = screen.get_rect().width/2 + screen.get_rect().height/2 * 1j

    segments = [Segment(50, 1, 0), Segment(200, 2, 0), Segment(100, 3, 0)]

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))

        current_pos = screen_mid
        for segment in segments:
            pygame.draw.circle(
                screen, (127, 127, 127),
                (numpy.real(current_pos), numpy.imag(current_pos)),
                segment.radius, width=1)
            current_pos += segment.get_vector()

        current_pos = screen_mid
        for segment in segments:
            end = current_pos + segment.get_vector()
            pygame.draw.line(
                screen, (255, 255, 255),
                (numpy.real(current_pos), numpy.imag(current_pos)),
                (numpy.real(end), numpy.imag(end)))
            current_pos += segment.get_vector()

        for segment in segments:
            segment.inc_angle(0.001)

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
