import numpy
import scipy
import math

from dataclasses import dataclass
import pygame
from svgpathtools import Path, svg2paths


SPEED = 0.001


@dataclass
class Segment:
    radius: float
    omega: float
    angle: float

    def get_vector(self):
        return self.radius * numpy.exp(2 * numpy.pi * 1j * self.angle)

    def inc_angle(self, dt):
        self.angle += self.omega * 2 * numpy.pi * dt


@dataclass
class NormalizedParametricEquation:
    equations: list[callable]

    def __call__(self, t):
        for equation in self.equations:
            try:
                return equation(t)
            except ValueError:
                continue


def lerp(a, b, t):
    return a + t * (b - a)


def bezier(start, end, control, t):
    return lerp(lerp(start, control, t), lerp(control, end, t), t)


def cubic_bezier(start, end, control1, control2, t):
    return lerp(bezier(start, control1, control2, t), bezier(control1, control2, end, t), t)


def path_element_to_equation(element):
    if isinstance(element, Path.Line):
        return lambda t: element.start + t * (element.end - element.start)
    elif isinstance(element, Path.CubicBezier):
        return lambda t: cubic_bezier(element.start, element.end, element.control1, element.control2, t)


def parse_svg(path):
    paths, _ = svg2paths(path)
    equations = [path_element_to_equation(element) for element in paths[0]]
    return NormalizedParametricEquation(equations)


def init_segments():
    segments = []
    for i in range(10):
        segments.append(Segment(50, 1, 0))
    return segments


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    screen_mid = screen.get_rect().width/2 + screen.get_rect().height/2 * 1j

    segments = init_segments()

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
            segment.inc_angle(1 * SPEED)

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
