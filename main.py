import numpy as np
import scipy as sp
import math

from numbers import Number

from dataclasses import dataclass
import pygame
from svgpathtools import svg2paths, Line, CubicBezier, QuadraticBezier


SPEED = 0.001


@dataclass
class Segment:
    radius: float
    omega: float
    angle: float

    def get_vector(self, t) -> complex:
        return self.radius * np.exp((math.tau * self.omega * t + self.angle)
                                    * 1j)


@dataclass
class FourierSeries:
    segments: list[Segment]

    def __call__(self, t) -> complex:  # TODO: change it to a method
        return sum(segment.get_vector(t) for segment in self.segments)


class NormalizedParametricEquation:
    def __init__(self, equations: list[callable], middle: complex) -> None:
        self.equations = equations
        self.middle = middle

        self.offset = self.middle - self._center()

    def _center(self) -> complex:
        return sp.integrate.quad(lambda t: self._start_at(t).real, 0, 1)[0] +\
            sp.integrate.quad(lambda t: self._start_at(t).imag, 0, 1)[0] * 1j

    def _start_at(self, t):
        i = math.floor(t * len(self.equations))
        try:
            return self.equations[i](t * len(self.equations) % 1)
        except IndexError:
            return 0 + 0j

    def at(self, t) -> Number:
        return self._start_at(t) + self.offset


def lerp(a, b, t) -> Number:
    return (1-t) * a + t * b


def bezier(start, end, control, t) -> Number:
    return lerp(lerp(start, control, t), lerp(control, end, t), t)


def cubic_bezier(start, end, control1, control2, t) -> Number:
    return lerp(bezier(start, control2, control1, t),
                bezier(control1, end, control2, t), t)


def path_element_to_equation(element) -> callable:
    if isinstance(element, Line):
        return lambda t: element.start + t * (element.end - element.start)
    elif isinstance(element, QuadraticBezier):
        return lambda t: bezier(element.start, element.end, element.control, t)
    elif isinstance(element, CubicBezier):
        return lambda t: cubic_bezier(element.start, element.end,
                                      element.control1, element.control2, t)


def parse_svg(path, wanted_middle) -> NormalizedParametricEquation:
    paths, _ = svg2paths(path)
    equations = [path_element_to_equation(element) for element in paths[0]]
    return NormalizedParametricEquation(equations, wanted_middle)


def init_segments(n: int, path: NormalizedParametricEquation) -> list[Segment]:
    segments = []

    def l(t, i):
        return (path.at(t) - path.middle) * np.exp(-i * math.tau * 1j * t)

    new_radius = sp.integrate.quad(lambda t: l(t, 0).real, 0, 1)[0] +\
                 sp.integrate.quad(lambda t: l(t, 0).imag, 0, 1)[0] * 1j
    segments.append(Segment(abs(new_radius), 0, np.angle(new_radius)))

    for i in range(1, n):
        new_radius = sp.integrate.quad(lambda t: l(t, i).real, 0, 1)[0] +\
                     sp.integrate.quad(lambda t: l(t, i).imag, 0, 1)[0] * 1j
        segments.append(Segment(abs(new_radius), i, np.angle(new_radius)))

        new_radius = sp.integrate.quad(lambda t: l(t, -i).real, 0, 1)[0] +\
                     sp.integrate.quad(lambda t: l(t, -i).imag, 0, 1)[0] * 1j
        segments.append(Segment(abs(new_radius), -i, np.angle(new_radius)))
    return segments


def dot(screen, pos: complex, size=1, color=(255, 0, 0)):
    pygame.draw.circle(
        screen, color,
        (np.real(pos), np.imag(pos)),
        size
    )


def main():  # TODO: make svg in middle
    # TODO: for the love of god, make this code more readable
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()

    font = pygame.font.Font('freesansbold.ttf', 32)

    # screen_mid = 0 + 0j
    # screen_mid = all_path.center()
    screen_mid = 400 + 300j

    all_path = parse_svg(r"forte-2-svgrepo-com.svg", screen_mid)

    path_points = [[all_path.at(p).real, all_path.at(p).imag]
                   for p in np.linspace(0, 0.999, 100)]

    segments = init_segments(50, all_path)
    series = FourierSeries(segments)

    t = 0
    while True:
        t = t % 1
        text = font.render(str(t - t % SPEED)[:4], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((0, 0, 0))

        screen.blit(text, text_rect)

        current_pos = screen_mid
        for segment in segments:
            pygame.draw.circle(
                screen, (127//2, 127//2, 127//2),
                (np.real(current_pos), np.imag(current_pos)),
                segment.radius, width=1)
            current_pos += segment.get_vector(t)

        current_pos = screen_mid
        for segment in segments:
            end = current_pos + segment.get_vector(t)
            pygame.draw.line(
                screen, (255, 255, 255),
                (np.real(current_pos), np.imag(current_pos)),
                (np.real(end), np.imag(end)))
            current_pos += segment.get_vector(t)

        pygame.draw.aalines(screen, (255/2, 0, 0), True, path_points)

        for g in np.linspace(t-0.1, t, 200):
            c = int(255 * (g - t + 0.1) / 0.1)
            dot(screen, series(g) + screen_mid, 1, (c, c, 0))

        pygame.display.flip()
        clock.tick(30)

        t += SPEED


if __name__ == "__main__":
    main()
