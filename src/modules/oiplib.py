import random
import math
import sympy as sp

class Particle:
    def __init__(
        self,
        func: function,
        x0: list,
        aCognitive: float = 2,
        aSocial: float = 2,
        inertia: float = 1,
    ):
        # The mathematical function to calculate the value for a given configuration.
        self.func: function = func

        # The number of parameters to be optimized.
        self.dimension: int = len(x0)

        # The coefficients that define particle dynamics.
        self.aCognitive: float = aCognitive
        self.aSocial: float = aSocial

        # Generate random initial velocities from 0 to 1.
        v0 = []
        for i in range(self.dimension):
            v0.append(random.uniform())

        # The current state of the system.
        self.x: list = x0
        self.v: list = v0
        self.inertia: float = inertia

        # The personal best.
        self.bestValue: float = self.fitness()
        self.bestPosition: list = self.x

    def update(self, globalBest: list):
        """Update the particle's state during each iteration.

        Args:

        globalBest (number list): Describes the configuration for the previously known global best.
        """
        try:
            vNext: list = []
            xNext: list = []

            for i in range(self.dimension):
                r1: float = random.uniform()
                r2: float = random.uniform()

                vNext.append(
                    self.inertia * self.v[i]
                    + self.aCognitive * (self.best[i] - self.x[i]) * r1
                    + self.aSocial * (globalBest[i] - self.x[i]) * r2
                )
                xNext.append(self.x[i] + vNext[i])

            self.x: list = xNext
            self.v: list = vNext

            currentFitness: float = self.func(*self.x)

            if currentFitness <= self.bestValue:
                self.bestValue: float = currentFitness
                self.bestPosition: list = self.x

        except IndexError:
            print(
                "WARN: Dimensions of global best must match amount of parameters to be optimized."
            )
            raise IndexError

    def fitness(self):
        """Returns the function value of the current position.
    
        Returns:

        fitness (float): The function value of the current position.
        """
        return self.func(*self.x)


class ParticleSwarm:
    def __init__(
        self,
        func: function,
        coordsMin: list,
        coordsMax: list,
        populationSize: int = 50,
        initStrategy: str = "random",
    ):
        try:
            # Define member variables.
            self.func: function = func
            self.dimensions: int = len(coordsMin)
            self.coordsMin: list = coordsMin
            self.coordsMin: list = coordsMax
            self.populationSize: int = populationSize
            self.particles: list = []

            # Particle instances of the particle swarm.
            for i in range(populationSize):
                x0 = [random.uniform() for i in range(self.dimensions)]
                self.particles.append(Particle(self.func, x0))

                if i == 0 or self.particles[i].bestValue < self.bestValue:
                    self.bestPosition = self.particles[i].bestPosition
                    self.bestValue = self.particles[i].bestValue

        except IndexError:
            print(
                "WARN: Dimensions of boundary minimum and boundary maximum must be the same."
            )
            raise IndexError

    def run(self, hysteresis: float = 1e-9, iterations: int = 100):
        bestValuePrevious = self.bestValue
        delta = hysteresis + 1
        iteration = 0

        while iteration < iterations:
            # Update the particle position and the particle velocities.
            for particle in self.particles:
                particle.update(self.bestPosition)

                # Check if the particle fitness is a new global best.
                if particle.bestValue <= self.bestValue:
                    self.bestValue = particle.bestValue
                    self.bestPosition = particle.bestPosition

            # Calculate the change in global best.
            delta = bestValuePrevious - self.bestValue

            # Check if error is within hysteresis.
            if delta <= hysteresis:
                iteration = iteration + 1
            else:
                iteration = 0


def ackley():
    x, y = sp.symbols("x y")
    f = sp.utilities.lambdify.implemented_function(
        "f",
        lambda x, y: math.exp(1)
        + 20
        - 20 * math.exp(-0.2 * math.sqrt(0.5 * (x * x + y * y)))
        - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))),
    )
    return sp.lambdify([x, y], f(x, y))

