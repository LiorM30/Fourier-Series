import numpy
import math

from numbers import Number

from dataclasses import dataclass
import pygame
from svgpathtools import Path, svg2paths, Line, CubicBezier


SPEED = 0.001


@dataclass
class Segment:
    radius: float
    omega: float
    angle: float

    def get_vector(self, t) -> complex:
        return self.radius * numpy.exp((math.tau * self.omega * t + self.angle)
                                       * 1j)


@dataclass
class NormalizedParametricEquation:
    equations: list[callable]

    def __call__(self, t) -> Number:
        try:
            print(self.equations[math.floor(t * len(self.equations))](t * len(self.equations) % 1))
            return self.equations[math.floor(t * len(self.equations))](t * len(self.equations) % 1)
        except IndexError:
            pass


def lerp(a, b, t) -> Number:
    return (1-t) * a + t * b


def bezier(start, end, control, t) -> Number:
    return lerp(lerp(start, control, t), lerp(control, end, t), t)


def cubic_bezier(start, end, control1, control2, t) -> Number:
    return lerp(bezier(start, control2, control1, t), bezier(control1, end, control2, t), t)


def path_element_to_equation(element) -> Number:
    if isinstance(element, Line):
        return lambda t: element.start + t * (element.end - element.start)
    elif isinstance(element, CubicBezier):
        return lambda t: cubic_bezier(element.start, element.end,
                                      element.control1, element.control2, t)


def parse_svg(path) -> NormalizedParametricEquation:
    paths, _ = svg2paths(path)
    equations = [path_element_to_equation(element) for element in paths[0]]
    return NormalizedParametricEquation(equations)


def init_segments(n: int) -> list[Segment]:
    segments = []
    for i in range(n):
        segments.append(Segment(50, (i / n), 0))
    return segments


def dot(screen, pos: complex, size=1):
    pygame.draw.circle(
        screen, (255, 0, 0),
        (numpy.real(pos), numpy.imag(pos)),
        size)


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    screen_mid = screen.get_rect().width/2 + screen.get_rect().height/2 * 1j

    font = pygame.font.Font('freesansbold.ttf', 32)

    segments = init_segments(10)
    all_path = parse_svg(r"C:\Users\User\Downloads\svg (1).svg")
    ps = [all_path(t) for t in numpy.linspace(0, 1, 1000)]
    t = 0
    while True:
        t = t % 1
        text = font.render(str(t - t % SPEED), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))

        screen.blit(text, text_rect)

        # current_pos = screen_mid
        # for segment in segments:
        #     pygame.draw.circle(
        #         screen, (127, 127, 127),
        #         (numpy.real(current_pos), numpy.imag(current_pos)),
        #         segment.radius, width=1)
        #     current_pos += segment.get_vector(t)

        # current_pos = screen_mid
        # for segment in segments:
        #     end = current_pos + segment.get_vector(t)
        #     pygame.draw.line(
        #         screen, (255, 255, 255),
        #         (numpy.real(current_pos), numpy.imag(current_pos)),
        #         (numpy.real(end), numpy.imag(end)))
        #     current_pos += segment.get_vector(t)

        p = all_path(t)
        dot(screen, p, 5)

        for p in ps:
            try:
                dot(screen, p)
            except TypeError:
                pass

        pygame.display.flip()
        clock.tick(30)

        t += SPEED


if __name__ == "__main__":
    main()
