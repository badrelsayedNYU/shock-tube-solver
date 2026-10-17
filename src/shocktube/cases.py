"""Riemann-problem test cases (spec §5). All use gamma = 1.4 on x in [0, 1]."""

from dataclasses import dataclass

from shocktube.gas import GAMMA


@dataclass(frozen=True)
class RiemannCase:
    name: str
    description: str
    WL: tuple[float, float, float]  # (rho, u, p) left of the diaphragm
    WR: tuple[float, float, float]  # (rho, u, p) right of the diaphragm
    x0: float
    t_end: float
    gamma: float = GAMMA


CASES = {
    "sod": RiemannCase(
        name="sod",
        description="Sod: shock, contact, rarefaction",
        WL=(1.0, 0.0, 1.0),
        WR=(0.125, 0.0, 0.1),
        x0=0.5,
        t_end=0.25,
    ),
    "123": RiemannCase(
        name="123",
        description='"123" double rarefaction (near vacuum)',
        WL=(1.0, -2.0, 0.4),
        WR=(1.0, 2.0, 0.4),
        x0=0.5,
        t_end=0.15,
    ),
    "blast": RiemannCase(
        name="blast",
        description="Strong blast (left half of Woodward-Colella)",
        WL=(1.0, 0.0, 1000.0),
        WR=(1.0, 0.0, 0.01),
        x0=0.5,
        t_end=0.012,
    ),
    "modsod": RiemannCase(
        name="modsod",
        description="Modified Sod (Toro): transonic rarefaction",
        WL=(1.0, 0.75, 1.0),
        WR=(0.125, 0.0, 0.1),
        x0=0.3,
        t_end=0.2,
    ),
}


def get_case(name):
    """Look up a test case by name ('sod', '123', 'blast', 'modsod')."""
    try:
        return CASES[name]
    except KeyError:
        raise ValueError(f"unknown case {name!r}; choose from {sorted(CASES)}") from None
