import random
import math
import sympy as sp


class Particle:
    def __init__(
        self,
        func,
        x0: list,
        aCognitive: float = 2,
        aSocial: float = 2,
        inertia: float = 0.5,
        dataset = None,
    ):
        # DEBUG
        self.coordinatesX = []
        self.coordinatesY = []
        self.coordinatesZ = []

        # The mathematical function to calculate the value for a given configuration.
        self.func = func
        
        # Dataset from which the fitness function compares func to
        self.dataset = dataset

        # The number of parameters to be optimized.
        self.dimension: int = len(x0)

        # The coefficients that define particle dynamics.
        self.aCognitive: float = aCognitive
        self.aSocial: float = aSocial

        # Generate random initial velocities from -1 to 1.
        v0 = []
        for i in range(self.dimension):
            v0.append(random.uniform(-1, 1))

        # The current state of the system.
        self.x: list = x0
        self.v: list = v0
        self.inertia: float = inertia

        # The personal best.
        if self.dataset is not None:
            self.bestValue: float = self.func(*self.x, self.dataset)
        else:
            self.bestValue: float = self.func(*self.x)
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
                r1: float = random.uniform(0, 1)
                r2: float = random.uniform(0, 1)

                vNext.append(
                    self.inertia * self.v[i]
                    + self.aCognitive * (self.bestPosition[i] - self.x[i]) * r1
                    + self.aSocial * (globalBest[i] - self.x[i]) * r2
                )
                xNext.append(self.x[i] + vNext[i])

            self.x: list = xNext
            self.v: list = vNext

            if self.dataset is not None:
                currentFitness: float = self.func(*self.x, self.dataset)
            else:
                currentFitness: float = self.func(*self.x)

            if currentFitness <= self.bestValue:
                self.bestValue: float = currentFitness
                self.bestPosition: list = self.x

            # DEBUG
            self.coordinatesX.append(self.bestPosition[0])
            self.coordinatesY.append(self.bestPosition[1])
            self.coordinatesZ.append(self.bestValue)

        except IndexError:
            print(
                "WARN: Dimensions of global best must match amount of parameters to be optimized."
            )
            raise IndexError


class ParticleSwarm:
    def __init__(
        self,
        func,
        coordsMin: list,
        coordsMax: list,
        populationSize: int = 50,
        initStrategy: str = "random",
        dataset = None,
    ):
        # DEBUG
        self.coordinatesX = []
        self.coordinatesY = []
        self.coordinatesZ = []
        try:
            # Define member variables.
            #x, y = sp.symbols("x y")
            #self.func = sp.lambdify((x, y), func)
            self.func = func
            self.dimensions: int = len(coordsMin)
            self.coordsMin: list = coordsMin
            self.coordsMax: list = coordsMax
            self.populationSize: int = populationSize
            self.particles: list = []
            self.dataset = dataset

            # Particle instances of the particle swarm.
            for i in range(populationSize):
                x0 = [
                    random.uniform(self.coordsMin[j], self.coordsMax[j])
                    for j in range(self.dimensions)
                ]

                self.particles.append(Particle(self.func, x0, dataset = self.dataset ))

                if i == 0 or self.particles[i].bestValue < self.bestValue:
                    self.bestPosition = self.particles[i].bestPosition
                    self.bestValue = self.particles[i].bestValue

        except IndexError:
            print(
                "WARN: Dimensions of boundary minimum and boundary maximum must be the same."
            )
            raise IndexError

    def run(self, hysteresis: float = 1e-6, iterations: int = 25 , print_best = False):
        delta = hysteresis + 1
        iteration = 0

        while iteration < iterations:
            bestValuePrevious = self.bestValue
            # Update the particle position and the particle velocities.
            for particle in self.particles:
                particle.update(self.bestPosition)

                # Check if the particle fitness is a new global best.
                if particle.bestValue <= self.bestValue:
                    self.bestValue = particle.bestValue
                    self.bestPosition = particle.bestPosition

            # Calculate the change in global best.
            try:
                delta = bestValuePrevious - self.bestValue
            except TypeError:
                delta = bestValuePrevious - self.bestValue[0]

            # Check if error is within hysteresis.
            if delta <= hysteresis:
                iteration = iteration + 1
            else:
                iteration = 0
                # DEBUG
                self.coordinatesX.append(self.bestPosition[0])
                self.coordinatesY.append(self.bestPosition[1])
                self.coordinatesZ.append(self.bestValue)
                if print_best == True:
                    print(self.bestValue)
            
            


def ackley():
    x, y = sp.symbols("x y")
    f = sp.Function("f")
    f = (
        sp.exp(1)
        + 20
        - 20 * sp.exp(-0.2 * sp.sqrt(0.5 * (x * x + y * y)))
        - sp.exp(0.5 * (sp.cos(2 * sp.pi * x) + sp.cos(2 * sp.pi * y))),
    )
    
    return f, x, y

f, x, y = ackley()
ackley_lambdified = sp.lambdify([x,y], f)

def ackley_function(x,y):
    return  ackley_lambdified(x,y)[0]

def fitness_function(a,b,dataset):
    fit = 0
    for i in range(len(dataset)):
        #x value
        x = dataset[i,0]

        #y value
        y = dataset[i,1]

        #z value
        z = dataset[i,2]
        
        r = z - ((a-x)**2 + b*(y-x**2)**2)
        
        fit = r**2 + fit
        
    return fit

