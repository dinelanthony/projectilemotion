"""
Objective: Using solve_ivp to model the path of a projectile
undergoing drag. Then, using this information find the
distance to impact, maximum height, time of flight,
and velocity at time of impact.
"""
"""
Operation: Under "Constants Definitions," the parameters of the function
may be changed to see how varying different initial conditions affects
the trajectory of the object. After making any changes, running the code
will show the updated trajectory of the object.
"""
"""
NOTE:

The projectile does not exactly hit 0 after being launched
according to the plot below. This is because the given
functions take discrete values of time to sample at,
which means that the height at the sampled time may
not exactly be 0. To achieve a result of an actual 0, we would
require an infinite number of time stamps (integration).
The plot below shows the best possible estimation using the
restriction of having ~20 points on the plot, as well as not
showing a negative height.
"""

# Import all necessary libraries to run code
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np


def derive(t, yArray):
    """
    Return the derivatives of position and velocity as components.

    t is the time stamp at which to take the derivative (not used)
    yArray is the array with the initial values of the position and velocity
    """
    vx = yArray[2]
    vy = yArray[3]
    ay = -g - c * vy / m
    ax = -c * vx / m
    return np.array([vx, vy, ax, ay])


def analyticalY(t):
    """"
    Returns the analytical value of the height of the projectile under drag.

    t is the time stamp at which to calculate the analytical height
    """
    return(vT/g*(vT+v0*np.sin(theta))*(1-np.exp(-g*t/vT))-vT*t)


def analyticalX(t):
    """
    Returns the analytical value of the distance of the projectile under drag.

    t is the time stamp at which to calculate the analytical distance.
    """
    return(v0*vT/g*(1-np.exp(-g*t/vT))*np.cos(theta))


def groundHit(t, yArray):
    """
    Returns the height of the projectile as an array.

    t is the time stamp at which to check the height (not used)
    yArray is the array with the initial values of position and velocity
    """
    return yArray[1]


# Parameters to terminate the solve_ivp function when y-position=0
groundHit.terminal = True
groundHit.direction = -1

# Constants Definitions
c = 0.65  # Drag constant in SI Units
g = 9.81
m = 0.1
v0 = 10
theta = np.deg2rad(50)
vT = m * g / c  # Terminal velocity in m/s
t0 = 0.00
tMax = 2 * v0 * np.sin(theta) / g  # Maximum time of flight in s

# Create an array of evenly spaced time stamps
timeSteps = 10000
tArray = np.linspace(t0, tMax, timeSteps)

# Initialize arrays to store the analytical values of distance and height
realX = np.zeros(len(tArray))
realY = np.zeros(len(tArray))

# Create an array of the starting conditions to feed into solve_ivp
initials = np.array([0, 0, v0 * np.cos(theta), v0 * np.sin(theta)])

# Fill the arrays with the analytical values of x and y in order to plot them
realX = analyticalX(tArray)
realY = analyticalY(tArray)

# IVP function
sol = integrate.solve_ivp(derive, (t0, tMax), initials,
                          t_eval=tArray, method='LSODA', events=groundHit)

# Store distance and height solutions in their respective places
numX = sol.y[0]
numY = sol.y[1]

# Make the numerical plot have 20 points
pack = np.round(np.linspace(0, len(numX)-1, 20)).astype(int)
realnumX = numX[pack]
realnumY = numY[pack]

# Plotting Functions
plt.style.use('dark_background')
plt.title("Trajectory of a Projectile Experiencing Air Resistance")
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.plot(realnumX, realnumY, 'ro', label='Numerical Path')
# Only consider positive height and position values
plt.plot(realX[np.where(realY >= 0)], realY[np.where(realY >= 0)],
         label='Analytical Path')
plt.legend(loc='upper left')

# Final Output Definitions
distance = sol.y[0][-1]
timeofFlight = max(sol.t)
maxHeight = max(sol.y[1])
vxImpact = sol.y[2][-1]
vyImpact = sol.y[3][-1]
vImpact = np.sqrt(vxImpact ** 2 + vyImpact ** 2)
thetaImpact = np.arctan(vyImpact / vxImpact)

# Final Output Print
print("Time of Flight (T): %.3fs\n"
      "Distance: %.3fm\n"
      "Maximum Height: %.3fm\n"
      "Speed at Impact Point: %.3fm/s\n"
      "Velocity at Impact Point: [%.3fm/s, %.3frad from the x-axis]"
      % (timeofFlight, distance, maxHeight, vImpact, vImpact, thetaImpact))