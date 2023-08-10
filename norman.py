# See https://pubmed.ncbi.nlm.nih.gov/908672/
# M = 1.5 W + 2.0 (W + L)(L/W)^2 + n(W + L)(1.5V^2 + 0.35VG)
#   M: Metabolic rate (watts)
#   W: Your weight (kg)
#   L: Weight of pack (kg)
#   V: Hiking speed (m/s)
#   G: Grade of incline (%)
#   n: Terrain factor
#     1.0 for paved road
#     1.1 for dirt road
#     1.2 for gravel road
#     1.3 for hard-packed snow
#     1.3+0.08/cm depression for soft snow
#     1.5 for heavy brush
#     1.6 for swampy bog
#     2.1 for loose sand

# Columns in Caltopo file:
#   0: Lat | 1: Lng | 2: Distance (meters) | 3: Distance (feet)
#   4: Distance (miles) | 5: Elevation (meters) | 6: Elevation (feet)
#   7: Slope (degrees) | 8: Aspect (degrees) | 9: Landcover
#   10: Canopy (percent)

# Methodology:
#   Calculate how long we will be hiking each segment
#   Integrate over each segment to get joules expended
#   Convert to calories

# Base calories:
# See https://pubmed.ncbi.nlm.nih.gov/2305711/

import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

def pandolf(w, l, v, g, n):
    return 1.5 * w + 2.0 * (w + l) * (l / w) ** 2 + n * (w + l) * (1.5 * v ** 2 + 0.35 * v * g)

def terrain_factors(terrain):
    factor_dict = {"Developed" : 1.0,
                   "Grassland" : 1.1,
                   "Barren"    : 1.1,
                   "Forest"    : 1.2,
                   "Shrub"     : 1.2,
                   "Crops"     : 1.2}
    factors = np.zeros_like(terrain)
    not_found = []
    for i in range(len(terrain)):
        try:
            factors[i] = factor_dict[terrain[i]]
        except:
            if terrain[i] not in not_found:
                print("\tterrain \"{}\" not found: using default value".format(terrain[i]))
                not_found.append(terrain[i])
            factors[i] = 1.2
    return factors

class RouteData:
    def __init__(self, fname):
        """Initialize the route.
        
        Args:
          fname: relative or absolute directory of a csv file from caltopo
        """
        self.f = pd.read_csv(fname)
        return
    def energy_expenditure(self,
                           weight,            # kg
                           base_weight,       # kg
                           consumable_weight, # kg
                           velocity,          # m/s
                           plot = False,
                           height = None,     # cm
                           gender = None,
                           age = None,
                           days = None):       # years
        """Calculate energy expenditure over entire route.
        
        Args: 
          weight: How much you weight! A constant.
          base_weight: Final weight of your pack at the end of the hike, including water.
          consumable_weight: Weight of consumables that will be gone by end end of the hike.
          velocity: Speed of hiking.
          height: Height.
          gender: For base calorie calculation.
          age: Age in years. 
        """
        distance = np.array(self.f["Distance (meters)"])
        elevation = np.array(self.f["Elevation (meters)"])
        grade = np.maximum(0.0, 100.0 * np.gradient(elevation, distance))
        terrain = self.f["Landcover"]
        terrain_factor = terrain_factors(terrain)
        time = distance / velocity
        pack_weight = np.linspace(base_weight + consumable_weight, base_weight, num=len(distance))
        wattage = pandolf(weight, pack_weight, velocity, grade, terrain_factor)
        joules = np.trapz(wattage, time)
        joules_per_calorie = 4184
        kcal = joules / joules_per_calorie
        print("calories over trip: {}".format(kcal))
        print("calories per hour: {}".format(kcal / (time[-1] / 3600)))

        kcal_base = 0.0
        if height > 0 and age > 0 and gender is not None:
            kcal_base = 10 * weight + 6.25 * height - 5 * age
            if gender == 'male':
                kcal_base += 5
            elif gender == 'female':
                kcal_base -= 161
            else:
                kcal_base -= 78
            print("base calories per day: {}".format(kcal_base))

        if days is not None:
            kcal_daily = (kcal_base * days + kcal) / days
            print("calories per day: {}".format(kcal_daily))
            
        if plot:
            plt.figure()
            plt.plot(time / 3600, elevation)
            plt.xlabel("time (hours)")
            plt.ylabel("elevation (meters)")

            import scipy.signal
            plt.figure()
            kcal_per_hour = scipy.signal.medfilt(wattage / joules_per_calorie * 3600, kernel_size=19)
            plt.plot(time / 3600, scipy.signal.medfilt(kcal_per_hour, kernel_size=19))
            plt.xlabel("time (hours)")
            plt.ylabel("kcal per hour")
            plt.ylim([0.0, np.max(kcal_per_hour)])
            plt.show()
        
        return
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weight', type=float, help='your weight (kg)', required=True)
    parser.add_argument('-b', '--base_weight', type=float, help='base weight (kg)', required=True)
    parser.add_argument('-c', '--consumable_weight', type=float, help='consumable weight (kg)', required=True)
    parser.add_argument('-v', '--velocity', type=float, help='velocity (m/s)', required=True)
    parser.add_argument('-f', '--file', type=str, help='csv file path from caltopo', required=True)
    parser.add_argument('-p', '--plot', action='store_true', help='plot some stuff')
    parser.add_argument('-t', '--height', type=float, default=0.0, help='height (cm), if you want to add in base calories')
    parser.add_argument('-a', '--age', type=float, default=0.0, help='age (years), if you want to add in base calories')
    parser.add_argument('-g', '--gender', choices=['male', 'female', 'other'], default=None, help='gender, if you want to add in base calories')
    parser.add_argument('-d', '--days', type=float, default=None, help='days, if you want an average daily expenditure')
    args = parser.parse_args()

    data = RouteData(args.file)
    data.energy_expenditure(args.weight,
                            args.base_weight,
                            args.consumable_weight,
                            args.velocity,
                            args.plot,
                            args.height,
                            args.gender,
                            args.age,
                            args.days)

