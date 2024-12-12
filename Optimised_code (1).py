# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 12:01:02 2024

@author: illsl
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:25:19 2024

@author: matthewillsley
"""


#######Import libraries and set up plotting defaults######
import random
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.animation import FuncAnimation
import pickle
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import time 

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

#######length and statistical parameters######
DAYS = 28
NUM_CYCLES = 12
TRIALS = 100 # You can increase the number of trials
cycle_lengths = [12]  # Cycle lengths to iterate over to test std-deviations

######Scale parameters######
MAX_BACTERIA_CAPACITY = 10000
# Set so bacteria can survive indefinetly without glycogen
HIBERNATION_TIME = DAYS * NUM_CYCLES
GLYCOGEN_SCALE = 100  # maximum number of new glycogen clumps generated per step
GLYCOGEN_CAPACITY = 6  # The number of bacteria that a glycogen cluster can support
NUMBER_IMMUNE_CELLS = 3  # scaling for number of immune cells made per step
INITIAL_GOOD = 150
INITIAL_BAD = 100
INITIAL_CA = 0
JENSII = True  # if want one species to dominate with CA125

#######Interaction Specifics for different species######
GLYCOGEN_INTERACTION_RANGE = 0.5  # How far bacteria can see glycogen
REPLICATE_PROB = 0.5

IMMUNE_CELL_INTERACTION_RANGE = 0.5  # How far immune cells can 'see' bacteria
PROB_KILL = 0.2  # Probability that once an immune cell 'sees' a bacteria that it kills the bacteria
NK_IMMUNE_CELL_INTERACTION_RANGE = 2 * IMMUNE_CELL_INTERACTION_RANGE
IMMUNE_CELL_AGE_LIMIT = DAYS * 2
# Limits to the number of bacteria one immune cell can kill, very sensitive to change
MACROPHAGE_KILL_LIMIT = 3
NEUTROPHIL_KILL_LIMIT = 2
NK_CELL_KILL_LIMIT = 2

#######Tumour production and parameteres######
CYCLE_START = 3
STEP_START = 1

GRADE_THRESHOLDS = {  # Thresholds in volume [cm^3] where cancer becomes more serious
    "grade_1": 0,
    "grade_2": 150,
    "grade_3": 300,
    "grade_4": 400
}

THRESHOLD_CA125 = 2  # number of ca125 to inhibit bacteria replication
HALF_LIFE = 10.4  # half life of ca125 marker, based on research
# number of ca125 there need to be for an inflammation marker
INFLAMMATION_THRESHOLD = 3


#######Heatmap parameter and size of the microbiome######
SIDE_LENGTH = 10
grid_size = 10  # 10x10 grid
bin_size = 2  # Each bin is 10x10 units
num_bins = grid_size // bin_size


#######Classes for Microbes and the environment######

class Microbe:
    def __init__(self, species, location, time_without_glycogen=0, inhibition=False):
        self.species = species
        self.location = location
        self.alive = True  # Track whether the microbe is alive
        # Indicates if the microbe is newly created
        self.time_without_glycogen = time_without_glycogen
        self.inhibition = inhibition

    def move(self):
        new_location = (self.location[0] + random.uniform(-SIDE_LENGTH * 0.1, SIDE_LENGTH * 0.1),
                        self.location[1] + random.uniform(-SIDE_LENGTH * 0.1, SIDE_LENGTH * 0.1))

        # Check if the new location is within bounds (0, SIDE_LENGTH)
        if new_location[0] < 0:
            new_location = (0, self.location[1])
        elif new_location[0] > SIDE_LENGTH:
            new_location = (SIDE_LENGTH, self.location[1])

        if new_location[1] < 0 or new_location[1] > SIDE_LENGTH:
            self.alive = False  # Mark microbe as dead
            return True  # Microbe has died (out of bounds)

        self.location = new_location  # Update location if valid
        return False  # Microbe is still alive

    def interact_with_glycogen(self, glycogen_objects):
        glycogen_consumed = False
        for glycogen in glycogen_objects[:]:  # Iterate over a copy of the list
            if calculate_distance(self.location, glycogen.location) <= GLYCOGEN_INTERACTION_RANGE:
                self.time_without_glycogen = 0  # Reset counter on glycogen consumption
                glycogen_consumed = True
                if glycogen.reduce_amount(1 / GLYCOGEN_CAPACITY):
                    # Remove depleted glycogen
                    glycogen_objects.remove(glycogen)
                break  # Stop looking once glycogen is consumed

        if not glycogen_consumed:
            self.time_without_glycogen += 1

        # Kill microbe if it has been without glycogen for too long
        if self.time_without_glycogen > HIBERNATION_TIME:
            self.alive = False

            return False
        return glycogen_consumed

    def replicate(self, glycogen_objects):
        if not self.alive:
            return None  # Dead microbes don't replicate

        nearby_glycogen = 0
        for glycogen in glycogen_objects:
            if calculate_distance(glycogen.location, self.location) <= GLYCOGEN_INTERACTION_RANGE:
                nearby_glycogen += glycogen.amount

        if nearby_glycogen > 1 and "Good_Bacteria" in self.species and random.uniform(0, 1) < REPLICATE_PROB:
            new_bacterium = Microbe(self.species, self.location)
            return new_bacterium
        elif nearby_glycogen > 1 and "Bad_Bacteria" in self.species and random.uniform(0, 1) < REPLICATE_PROB:
            new_bacterium = Microbe(self.species, self.location)
            return new_bacterium

        else:

            return None  # No replication if no glycogen is nearby

    def inhibit(self, ca125_objects):
        if calculate_local_concentration(self.location, ca125_objects) > THRESHOLD_CA125:
            self.inhibition = True
        else:
            self.inhibition = False

    def interact_with_bacteriocins(self, bacteriocin_objects, bacteria):
        bacteriocin = next((x for x in bacteriocin_objects if calculate_distance(
            x.location, self.location) <= 0.1), None)
        if bacteriocin != None:
            bacteria.remove(self)
            bacteriocin_objects.remove(bacteriocin)


class Tumour:
    def __init__(self, subtype, grade, stage, size, asymptomatic=True, time=0):
        self.subtype = subtype
        self.grade = grade
        self.stage = stage
        self.size = size
        self.time = time
        self.asymptomatic = asymptomatic

    def secrete_ca125(self, ca125_objects):
        # Create a new list for the current level of CA125 secretion
        produced = []

        # Determine the CA125 secretion level based on tumour grade and stage
        if "grade_4" in self.grade:
            ca125_level = random.randint(4, 7)
        elif "grade_3" in self.grade:
            ca125_level = random.randint(3, 5)
        elif "grade_2" in self.grade:
            ca125_level = random.randint(1, 3)
        elif "grade_1" in self.grade:
            ca125_level = random.randint(0, 2)
        else:
            ca125_level = 0

        if self.stage == 4:
            ca125_level += random.randint(3, 5)
        elif self.stage == 3:
            ca125_level += random.randint(2, 3)
        elif self.stage == 2:
            ca125_level += random.randint(1, 2)
        elif self.stage == 1:
            ca125_level += random.randint(0, 1)
        else:
            ca125_level += 0

        # Create new CA125 objects for this time step
        #if self.asymptomatic == False:
        for i in range(ca125_level):
            ca125_object = Protein("CA125",
                                   location=(random.uniform(0, SIDE_LENGTH),
                                             random.uniform(9, 10))
                                   )
            ca125_objects.append(ca125_object)
        return ca125_objects
        # Return the current secretion level as a list

    def secrete_vegf(self):
        if self.size > 0.008:
            vegf_concentration = random.uniform(0.01, 0.03)
        elif self.size > 300:
            vegf_concentration = random.uniform(0.3, 3)
        else:
            vegf_concentration = 0
        return vegf_concentration

    def increase_size(self, tumour_sizes):

        vegf_factor = 0

        if self.subtype == "high_grade":
            growth_rate = 0.02
        # Slower growth for low-grade tumours
        elif self.subtype == "low_grade":
            growth_rate = 0.01  # Gentler growth rate

        eff_growth_rate = vegf_factor + growth_rate

        V_0 = 1
        V_max = 520

        log_bracket = np.log(V_0/V_max)
        exponential = math.exp(-eff_growth_rate*self.time)

        self.size = V_max*math.exp(log_bracket*exponential)

        tumour_sizes.append(self.size)

    def grade_change(self):
        if self.size > 400:
            self.grade = "grade_4"

        elif self.size > 300:
            self.grade = "grade_3"

        elif self.size > 150:
            self.grade = "grade_2"
        else:
            self.grade = "grade_1"

    def stage_change(self, VEGF):
        previous_stage = self.stage

        # Update stage based on VEGF levels
        if VEGF > 1 and self.stage < 4:
            self.stage = 4
        elif VEGF > 0.5 and self.stage < 3:
            self.stage = 3
        elif VEGF > 0.02 and self.stage < 2:
            self.stage = 2
        elif VEGF > 0.01 and self.stage < 1:
            self.stage = 1

        if self.stage > previous_stage:
            if self.asymptomatic:

                if self.stage == 1:
                    probability = 0.5
                elif self.stage == 2:
                    probability = 0.5
                elif self.stage == 3:
                    probability = 0.8
                elif self.stage == 4:
                    probability = 0.9
                else:
                    probability = 0

                # Determine if the tumor becomes symptomatic
                if random.random() < probability:
                    self.asymptomatic = False  # Tumor becomes symptomatic
                    print('tumour is symtomatic')

    def age_tumour(self):
        self.time += 1


class Protein:
    def __init__(self, species, location, alive=True):
        self.species = species
        self.location = location
        self.alive = True

    def move(self, ca125_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.1

        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('inf')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, ca125_objects)
                if concentration < min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0 or self.location[1] > SIDE_LENGTH:
            self.alive = False  # Mark as dead if out of bounds
            return True  # Glycogen has moved out of bounds and is considered "dead"

        return False  # Glycogen is still alive


class Glycogen:
    def __init__(self, location, amount=1.0, alive=True):
        self.location = location  # (x, y) coordinates
        self.amount = amount  # Amount of glycogen in the clump
        self.alive = True

    def reduce_amount(self, amount):
        """Reduce the glycogen amount when bacteria consume it."""
        self.amount -= amount
        if self.amount <= 0:
            self.amount = 0  # Ensure the amount never goes below 0
            return True  # Return True if the glycogen is depleted
        return False  # Return False if there is still glycogen left

    def move(self, glycogen_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.1
        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('inf')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, glycogen_objects)
                if concentration < min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0 or self.location[1] > SIDE_LENGTH:
            self.alive = False  # Mark as dead if out of bounds
            return True  # Glycogen has moved out of bounds and is considered "dead"

        return False  # Glycogen is still alive
# Immune Cell Class


class ImmuneCell:
    def __init__(self, location, age=0):
        self.location = location
        self.age = age

    def move(self, bacteria_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.1

        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        max_concentration = 0
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, bacteria_objects)
                if concentration > max_concentration:
                    max_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0 or self.location[1] > SIDE_LENGTH:
            self.alive = False  # Mark as dead if out of bounds
            return True  # Glycogen has moved out of bounds and is considered "dead"

        return False  # Glycogen is still alive

    def age_cell(self):
        # General aging behavior for all immune cells
        self.age += 1
        if self.age > IMMUNE_CELL_AGE_LIMIT:

            return True  # Return True if the cell dies of old age
        return False


class Neutrophil(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)  # Initialize the base class (ImmuneCell)
        self.kill_count = kill_count

    def interact(self, all_bacteria):
        # Macrophages can kill both good and bad bacteria in their proximity
        killed = False
        # Kill good bacteria if nearby

        for bacterium in all_bacteria:
            if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE\
                    and PROB_KILL > random.uniform(0, 1):
                bacterium.alive = False
                all_bacteria.remove(bacterium)
                self.kill_count += 1

                if self.kill_count >= NEUTROPHIL_KILL_LIMIT:
                    killed = True
        return killed, all_bacteria


class Macrophage(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)
        self.kill_count = kill_count  # Track how many bacteria the macrophage has killed

    def interact(self, all_bacteria):
        killed = False
        for bacterium in all_bacteria:
            if calculate_distance(self.location, bacterium.location) <= IMMUNE_CELL_INTERACTION_RANGE \
                    and PROB_KILL < random.uniform(0, 1):
                bacterium.alive = False
                all_bacteria.remove(bacterium)
                self.kill_count += 1
                # engulf if close bacteria
                another_bacterium = next((x for x in all_bacteria if calculate_distance(
                    x.location, bacterium.location) <= 0.01), None)
                if another_bacterium != None and another_bacterium != bacterium:
                    another_bacterium.alive = False
                    all_bacteria.remove(another_bacterium)
                    print("Engulfed!")
                    self.kill_count += 1

                if self.kill_count >= MACROPHAGE_KILL_LIMIT:
                    killed = True
        return killed, all_bacteria


class NKcell(ImmuneCell):
    def __init__(self, location, age=0, kill_count=0):
        super().__init__(location, age)  # Initialize base class (ImmuneCell)
        self.kill_count = kill_count  # Track how many bacteria the NK cell has killed

    def interact(self, all_bacteria):
        killed = False
        for bacterium in all_bacteria:
            if "Bad_Bacteria" in bacterium.species:
                if calculate_distance(self.location, bacterium.location) <= NK_IMMUNE_CELL_INTERACTION_RANGE \
                        and PROB_KILL < random.uniform(0, 1):
                    bacterium.alive = False
                    all_bacteria.remove(bacterium)
                    self.kill_count += 1

                    if self.kill_count >= NK_CELL_KILL_LIMIT:
                        killed = True
        return killed, all_bacteria


class Inflammation:
    def __init__(self, location):
        self.location = location

    def immune_reaction(self, location, inflammation_objects, immune_cells):
        if calculate_local_concentration(self.location, inflammation_objects) > INFLAMMATION_THRESHOLD:
            immune_cell_responce(location, immune_cells)

    def move(self, inflammation_objects):
        # Define small movement step size
        step_size = SIDE_LENGTH * 0.2
        # Check concentration in adjacent directions
        directions = [
            (step_size, 0),  # Right
            (-step_size, 0),  # Left
            (0, step_size),  # Up
            (0, -step_size)  # Down
        ]

        min_concentration = float('inf')
        best_direction = (0, 0)

        for dx, dy in directions:
            new_location = (self.location[0] + dx, self.location[1] + dy)
            if 0 <= new_location[0] <= SIDE_LENGTH and 0 <= new_location[1] <= SIDE_LENGTH:
                concentration = calculate_local_concentration(
                    new_location, inflammation_objects)
                if concentration < min_concentration:
                    min_concentration = concentration
                    best_direction = (dx, dy)

        # Move in the best direction with the lowest concentration
        self.location = (
            self.location[0] + best_direction[0], self.location[1] + best_direction[1])

        # Boundary check to ensure it stays within valid space
        if self.location[0] < 0:
            self.location = (0, self.location[1])
        elif self.location[0] > SIDE_LENGTH:
            self.location = (SIDE_LENGTH, self.location[1])

        if self.location[1] < 0 or self.location[1] > SIDE_LENGTH:
            self.alive = False  # Mark as dead if out of bounds
            return True  # Glycogen has moved out of bounds and is considered "dead"

        return False  # Glycogen is still alive


#######Standalone functions for simulation######


def read_data():
    hormone_data = np.genfromtxt("TabHormone.csv", comments='%',
                                 delimiter=",", skip_header=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7))
    day_no = hormone_data[:, 0]
    est_level = hormone_data[:, 2]
    progest_level = hormone_data[:, 5]
    return day_no, est_level, progest_level


def initialize_state(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count,
                     glycogen_objects, ca125_objects, inflammation_objects, tumour, step, pH_level):

    # Produce glycogen and retain existing objects
    _, glycogen_objects = glycogen_production(
        estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count,
        MAX_BACTERIA_CAPACITY, glycogen_objects)

    # Generate natural CA-125 levels
    natural_ca125_objects = natural_ca125(estrogen_level, progesterone_level)

    # If a tumor is present, combine natural and tumor-secreted CA-125
    if tumour is not None:
        tumour_ca125_objects = tumour.secrete_ca125(ca125_objects)
        ca125_objects = natural_ca125_objects + tumour_ca125_objects
        ca125_levels = len(ca125_objects)
        VEGF_levels = tumour.secrete_vegf()
    else:
        # Only natural CA-125 levels contribute if no tumor is present
        ca125_objects = natural_ca125_objects
        ca125_levels = len(ca125_objects)
        VEGF_levels = 0

    # Create inflammation objects based on CA-125 levels
    inflammation_objects = inflammation_creation(
        ca125_objects, inflammation_objects)

    # Construct and return the current state
    state = {
        'estrogen_level': estrogen_level,
        'progesterone_level': progesterone_level,
        'iron_level': iron_pulse(estrogen_level, progesterone_level),
        'pH': pH(pH_level, good_bacteria_count, bad_bacteria_count, step),
        'glycogen_objects': glycogen_objects,
        'glycogen_level': len(glycogen_objects),
        'cytokine_level': cytokine_level(good_bacteria_count, bad_bacteria_count),
        'ca125_objects': ca125_objects,
        'ca125_levels': ca125_levels,
        'inflammation_objects': inflammation_objects,
        "VEGF_level": VEGF_levels
    }
    return state

# Iron pulse function


def iron_pulse(estrogen_level, progesterone_level):
    if estrogen_level < 0.2 and progesterone_level < 0.2:
        iron_level = random.uniform(0.7, 1)
    else:
        iron_level = random.uniform(0.1, 0.3)
    return iron_level


def solution_pH_calc(pH1, pH2, pH3, vol1, vol2, vol3):
    M1 = 10**-pH1
    # print(pH2)
    M2 = 10**-pH2
    M3 = 10**-pH3
    mol1 = M1*vol1
    mol2 = M2*vol2
    mol3 = M3*vol3
    mol = mol1 + mol2 + mol3
    M = mol / (vol1+vol2+vol3)
    pH = -np.log10(M)
    return pH


def pH(pH, total_good_bacteria, total_bad_bacteria, step):
    # during menstruation menstrual blood reduces
   # print(environment["pH"])
    if step < 7:
        # pH blood
        pH_b = random.uniform(7.3, 7.5)
        if len(pH) == 0:
            # pH vagina
            pH_v = random.uniform(3.5, 4.5)
            vol_s = 0.05*10**-3

        else:
            total_bacteria = total_good_bacteria + total_bad_bacteria
            bacteria_proportion = total_good_bacteria/total_bacteria
            # initial pH
            pH_v = pH[-1]
            vol_s = 0.5*bacteria_proportion*10**-3

        # pH of secretion and volumes
        pH_s = random.uniform(3.5, 4.5)
        vol_b = 1*10**-3
        vol_v = 0.4*10**-3

    else:
        total_bacteria = total_good_bacteria + total_bad_bacteria
        bacteria_proportion = total_good_bacteria/total_bacteria
        pH_v = pH[-1]
        pH_s = random.uniform(3.5, 4.5)
        vol_v = 1*10**-3
        vol_s = 0.5*bacteria_proportion*10**-3

        pH_b = 0
        vol_b = 0

    new_pH = solution_pH_calc(pH_b, pH_v, pH_s, vol_b, vol_v, vol_s)
    new_pH = new_pH + random.uniform(-0.2, 0.2)
    return new_pH


def calculate_local_concentration(location, objects, radius=1):
    count = 0
    for obj in objects:
        if obj != location and calculate_distance(location, obj.location) <= radius:
            count += 1
    return count


def cytokine_level(good_bacteria_count, bad_bacteria_count):
    total_bacteria = good_bacteria_count + bad_bacteria_count
    try:
        bad_bacteria_proportion = bad_bacteria_count/total_bacteria
        if bad_bacteria_proportion > 0.8 or bad_bacteria_proportion < 0.2:
            return random.uniform(0.8, 0.9)
        elif bad_bacteria_proportion > 0.5:
            return random.uniform(0.3, 0.7)
        else:
            return random.uniform(0, 0.3)
    except:
        return random.uniform(0, .2)


def glycogen_production(estrogen_level, progesterone_level, good_bacteria_count, bad_bacteria_count, MAX_BACTERIA_CAPACITY, glycogen_objects):
    hormone_level = (estrogen_level + progesterone_level) / 2
    total_bacteria_count = good_bacteria_count + bad_bacteria_count
    capacity_usage = total_bacteria_count / MAX_BACTERIA_CAPACITY
    glycogen_count = 0

    # Avoid glycogen production if capacity is exceeded
    if capacity_usage >= 1:
        return glycogen_count, glycogen_objects

    try:
        # Calculate the number of glycogen clumps based on hormone levels and capacity usage
        clump_number = int(hormone_level * GLYCOGEN_SCALE)
        for _ in range(clump_number):
            glycogen = Glycogen(location=(random.uniform(
                0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            glycogen_objects.append(glycogen)
            glycogen_count += 1
    except Exception as e:
        print(f"Error in glycogen production: {e}")
        # Default to producing at least one glycogen object if something goes wrong
        glycogen = Glycogen(location=(random.uniform(
            0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
        glycogen_objects.append(glycogen)
        glycogen_count += 1

    return glycogen_count, glycogen_objects


def move_glycogen(glycogen_objects):
    for glycogen in glycogen_objects:
        glycogen.move(glycogen_objects)
        if glycogen.alive == False:
            glycogen_objects.remove(glycogen)


# Function to update the chemical levels in current_state


def update_chemical_levels(environment, state, good_bacteria_count, bad_bacteria_count, step):
    state['iron_level'] = iron_pulse(
        state['estrogen_level'], state['progesterone_level'])

    state['glycogen_level'] = len(state['glycogen_objects'])

    state['pH'] = environment["pH"]
    state['cytokine_level'] = cytokine_level(
        good_bacteria_count, bad_bacteria_count)
    return state


def initialize_bacteria(microbe_trackers, num_good_bacteria, num_bad_bacteria):
    """
    Initialize the microbe_trackers with a starting number of good and bad bacteria.
    """
    # Initialize good bacteria
    for i in range(1, 5):  # Assuming you have 4 types of good bacteria
        species_name = f"Good_Bacteria_{i}"
        for _ in range(num_good_bacteria):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species=species_name,
                              location=location)
            microbe_trackers['good object'][species_name].append(microbe)

    # Initialize bad bacteria
    for i in range(1, 3):  # Assuming you have 2 types of bad bacteria
        species_name = f"Bad_Bacteria_{i}"
        for _ in range(num_bad_bacteria):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species=species_name,
                              location=location)
            microbe_trackers['bad object'][species_name].append(microbe)

# Function to handle microbe interactions


def replenish_bacteria(microbe_trackers, species):
    if "Good_Bacteria" in species:

        for i in range(50):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species, location)
            microbe_trackers['good object'][species].append(microbe)
    elif "Bad_Bacteria" in species:

        for i in range(50):
            location = (random.uniform(0, SIDE_LENGTH),
                        random.uniform(0, SIDE_LENGTH))
            microbe = Microbe(species, location)
            microbe_trackers['bad object'][species].append(microbe)


def combine_bacteria(microbe_trackers):
    """
    Combine all bacteria from different arrays and dictionaries into a single list.
    """
    combined_bacteria = []

    # Combine good bacteria from all species in "good object"
    for good_species, good_bacteria_list in microbe_trackers["good object"].items():
        combined_bacteria.extend(good_bacteria_list)

    # Combine bad bacteria from all species in "bad object"
    for bad_species, bad_bacteria_list in microbe_trackers["bad object"].items():
        combined_bacteria.extend(bad_bacteria_list)

    return combined_bacteria  # Return the combined list of all bacteria


def add_bacteria(new_bacterium, glycogen_objects, new_bacteria):
    # Only proceed if a new bacterium is successfully replicated
    if new_bacterium is not None and new_bacterium.species is not None:

        new_bacterium.interact_with_glycogen(glycogen_objects)
        new_bacterium.move()  # New bacterium moves after replication
        # Add new bacteria to the list
        new_bacteria.append(new_bacterium)


def handle_microbes(state, microbe_trackers, glycogen_objects, bacteriocin_objects, bacteriocin_objects_E):
    """
    Process all bacteria by interacting with glycogen, moving, and attempting replication.
    Newly replicated bacteria are added to the appropriate species tracker.
    """

    if not any(microbe_trackers['good object'].values()) or not any(microbe_trackers['bad object'].values()):
        # Initialize with some starting bacteria
        initialize_bacteria(
            microbe_trackers, INITIAL_GOOD, INITIAL_BAD)

    # bacteriocins
    for bacteria in microbe_trackers["good object"]["Good_Bacteria_2"]:
        if random.randint(1, 3) == 1:
            bacteriocin = Protein("bacteriocin", bacteria.location)
            bacteriocin_objects.append(bacteriocin)

    for bacteria in microbe_trackers["good object"]["Good_Bacteria_3"]:
        if random.randint(1, 3) == 1:
            bacteriocin = Protein("bacteriocin", bacteria.location)
            bacteriocin_objects_E.append(bacteriocin)

    # Combine all existing bacteria into one list
    all_bacteria = combine_bacteria(microbe_trackers)

    # Shuffle the list for randomness in processing order
    random.shuffle(all_bacteria)
    try:

        good_bacteria_count = sum(microbe_trackers["good bacteria tracker"][species][-1]
                                  for species in microbe_trackers["good bacteria tracker"])
        bad_bacteria_count = sum(microbe_trackers["bad bacteria tracker"][species][-1]
                                 for species in microbe_trackers["bad bacteria tracker"])
        total_bacteria_count = good_bacteria_count + bad_bacteria_count
        good_proportion = good_bacteria_count/total_bacteria_count
    except:
        good_proportion = 0.5
    # Store new bacteria that will be added after processing
    new_bacteria = []
    replication_counter = 0
    # Process each bacterium
    move_glycogen(glycogen_objects)

    for bacterium in all_bacteria:
        if bacterium.alive:
            # Bacteria consume glycogen if nearby
            bacterium.interact_with_glycogen(glycogen_objects)
            bacterium.move()  # Bacteria move within the environment
            bacterium.inhibit(state["ca125_objects"])

            if replication_counter <= len(glycogen_objects):

                if ("Good_Bacteria" in bacterium.species and good_proportion < random.uniform(0.5, 1)) and bacterium.inhibition == False:

                    if state['pH'] < 4.5 and state['pH'] > 3.5:
                        # Attempt replication based on nearby glycogen
                        new_bacterium = bacterium.replicate(glycogen_objects)
                        add_bacteria(
                            new_bacterium, glycogen_objects, new_bacteria)
                        replication_counter += 1
                    else:
                        continue
                elif "Bad_Bacteria" in bacterium.species and good_proportion > random.uniform(0, 0.2):
                    # print(state["pH"])
                    if state['pH'] > 4:
                        replication_factor = random.randint(1, 2)
                        for i in range(replication_factor):
                            new_bacterium = bacterium.replicate(
                                glycogen_objects)
                            add_bacteria(
                                new_bacterium, glycogen_objects, new_bacteria)
                            replication_counter += 1

                    else:
                        # inhibit or no growth
                        number_generated = random.randint(0, 1)
                        if number_generated == 0:
                            continue
                        else:
                            for i in range(number_generated):
                                new_bacterium = bacterium.replicate(
                                    glycogen_objects)
                                add_bacteria(
                                    new_bacterium, glycogen_objects, new_bacteria)
                                replication_counter += 1

    # After all bacteria are processed, add the new bacteria to the trackers
    for new_bacterium in new_bacteria:
        if "Good_Bacteria" in new_bacterium.species:
            microbe_trackers['good object'][new_bacterium.species].append(
                new_bacterium)

        elif "Bad_Bacteria" in new_bacterium.species:
            microbe_trackers['bad object'][new_bacterium.species].append(
                new_bacterium)

    # Remove dead microbes from the trackers
    for good_species in microbe_trackers["good object"]:
        microbe_trackers["good object"][good_species] = [
            m for m in microbe_trackers["good object"][good_species] if m.alive]

    for bad_species in microbe_trackers["bad object"]:
        microbe_trackers["bad object"][bad_species] = [
            m for m in microbe_trackers["bad object"][bad_species] if m.alive]

    for good_species in microbe_trackers["good object"]:
        if not microbe_trackers["good object"][good_species]:
            # If the population of this good species is zero, replenish it
            print(f"Replenishing {good_species} population.")
            replenish_bacteria(microbe_trackers, good_species)

    for bad_species in microbe_trackers["bad object"]:
        if not microbe_trackers["bad object"][bad_species]:
            # If the population of this bad species is zero, replenish it
            print(f"Replenishing {bad_species} population.")
            replenish_bacteria(microbe_trackers, bad_species)
    # bacteriocins with their respective bacteria
    for microbe in microbe_trackers["bad object"]["Bad_Bacteria_1"]:
        microbe.interact_with_bacteriocins(
            bacteriocin_objects, microbe_trackers["bad object"]["Bad_Bacteria_1"])
    for microbe in microbe_trackers["bad object"]["Bad_Bacteria_2"]:
        microbe.interact_with_bacteriocins(
            bacteriocin_objects_E, microbe_trackers["bad object"]["Bad_Bacteria_2"])


def generate_immune_cells(state, microbe_trackers):
    if state['estrogen_level'] > 0.6:
        no_new_immune_cells = int((NUMBER_IMMUNE_CELLS + 2) *
                                  (1+state['cytokine_level']))
        for _ in range(no_new_immune_cells):
            immune_type = random.choice(["macrophage", "neutrophil", "NK"])
            if immune_type == "macrophage":
                new_immune_cell = Macrophage(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "neutrophil":
                new_immune_cell = Neutrophil(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "NK":
                new_immune_cell = NKcell(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            microbe_trackers['immune cells'].append(
                new_immune_cell)
    elif state["progesterone_level"] > 0.6:
        pass

    elif state["estrogen_level"] > 0.3:
        no_new_immune_cells = int(NUMBER_IMMUNE_CELLS *
                                  (1+state['cytokine_level']))
        for _ in range(no_new_immune_cells):
            immune_type = random.choice(["macrophage", "neutrophil", "NK"])
            if immune_type == "macrophage":
                new_immune_cell = Macrophage(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "neutrophil":
                new_immune_cell = Neutrophil(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))
            elif immune_type == "NK":
                new_immune_cell = NKcell(location=(random.uniform(
                    0, SIDE_LENGTH), random.uniform(0, SIDE_LENGTH)))

            microbe_trackers['immune cells'].append(
                new_immune_cell)
    microbe_trackers['immune production'].append(
        len(microbe_trackers['immune cells']))


# Function to handle immune cell interactions


def calculate_distance(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def handle_immune_cells(microbe_trackers):

    all_bacteria = combine_bacteria(microbe_trackers)
    microbe_trackers['good object'] = {
        f"Good_Bacteria_{i}": [] for i in range(1, 5)}
    microbe_trackers['bad object'] = {
        f"Bad_Bacteria_{i}": [] for i in range(1, 3)}

    # Shuffle the list for randomness in processing order
    random.shuffle(all_bacteria)

    for immune_cell in microbe_trackers['immune cells'][:]:
        killed = False

        if isinstance(immune_cell, Neutrophil):
            killed, bacteria_list = immune_cell.interact(all_bacteria)

        elif isinstance(immune_cell, Macrophage):
            killed, bacteria_list = immune_cell.interact(all_bacteria)

        elif isinstance(immune_cell, NKcell):
            killed, all_bacteria = immune_cell.interact(all_bacteria)

        if killed or immune_cell.age_cell() or immune_cell.move(all_bacteria):
            microbe_trackers['immune cells'].remove(immune_cell)

    for bacterium in all_bacteria:
        if "Good_Bacteria" in bacterium.species:
            microbe_trackers['good object'][bacterium.species].append(
                bacterium)
        if "Bad_Bacteria" in bacterium.species:
            microbe_trackers['bad object'][bacterium.species].append(bacterium)

    microbe_trackers['immune tracker'].append(
        len(microbe_trackers['immune cells']))

# Function to append the current step counts to trackers


def create_tumour(cycle, step):
    # Function to create a tumour at a specified time in the simulation
    if cycle == CYCLE_START and step == STEP_START:
        subtype = random.choice(['high_grade'])
        tumour = Tumour(subtype, grade='grade_1', stage=1,
                        size=0, asymptomatic=True)
        return tumour
    else:
        return None


def tumour_interaction(tumour, ca125_levels, tumour_sizes, VEGF):
    # Append the current secretion level (as a nested list) to ca125_levels

    # Call other tumour-related functions
    tumour.age_tumour()
    tumour.increase_size(tumour_sizes)
    tumour.grade_change()
    tumour.stage_change(VEGF[-1])


def immune_cell_responce(location, immune_cells):

    new_immune_cell = Neutrophil(location)
    immune_cells.append(new_immune_cell)


def ca125_interact(ca125_objects, good_bacteria_objects, immune_cell_objects):
    for ca in ca125_objects:
        ca.move(ca125_objects)
        ca125_decay(ca, ca125_objects)


def inflammation_creation(ca125_objects, inflammation_objects):
    # Initialize a grid to calculate local CA125 concentrations
    local_ca125_conc = np.zeros((num_bins, num_bins))
    present_state = []

    # Bin CA125 objects to calculate local concentration
    for ca in ca125_objects:
        x_bin, y_bin = get_bin(ca.location)
        local_ca125_conc[y_bin, x_bin] += 1

    # Convert grid indices to spatial coordinates
    bin_width = grid_size / num_bins

    # Check each grid point for CA125 threshold exceedance
    for y_bin in range(num_bins):
        for x_bin in range(num_bins):
            concentration = local_ca125_conc[y_bin, x_bin]

            if concentration > THRESHOLD_CA125:
                # Convert grid indices to spatial coordinates
                x_coord = (x_bin + 0.5) * bin_width  # Center of the bin
                y_coord = (y_bin + 0.5) * bin_width  # Center of the bin

                # Create inflammation marker at the correct spatial location
                inflammation_marker = Inflammation((x_coord, y_coord))
                present_state.append(inflammation_marker)

    return present_state


def handle_inflamation_markers(inflammation_objects, immune_cell_objects):

    for obj in inflammation_objects:

        obj.immune_reaction(
            obj.location, inflammation_objects, immune_cell_objects)


def ca125_decay(ca, ca_objects):
    prob_decay = math.log(2)/HALF_LIFE
    if random.uniform(0, 1) < prob_decay:
        ca_objects.remove(ca)


def natural_ca125(estrogen_level, progesterone_level):
    natural_objects = []
    if estrogen_level > 0.5 and progesterone_level > 0.5:
        number = random.randint(1, 3)
    elif estrogen_level < 0.5 or progesterone_level < 0.5:
        number = random.randint(0, 3)

    for i in range(number):
        ca125_object = Protein("CA125",
                               location=(random.uniform(0, SIDE_LENGTH),
                                         random.uniform(0, SIDE_LENGTH))
                               )
        natural_objects.append(ca125_object)
    return natural_objects


def update_trackers(microbe_trackers):
    # Update good bacteria tracker based on the lengths of the good object lists
    for good_species in microbe_trackers["good bacteria tracker"]:
        good_bacteria_count = len(
            microbe_trackers["good object"][good_species])
        microbe_trackers["good bacteria tracker"][good_species].append(
            good_bacteria_count)

    # Update bad bacteria tracker based on the lengths of the bad object lists
    for bad_species in microbe_trackers["bad bacteria tracker"]:
        bad_bacteria_count = len(microbe_trackers["bad object"][bad_species])
        microbe_trackers["bad bacteria tracker"][bad_species].append(
            bad_bacteria_count)

    # Calculate step-based plot data for good bacteria
    for good_species in microbe_trackers["good bacteria step plot"]:
        microbe_trackers["good bacteria step plot"][good_species] = [
            microbe_trackers["good bacteria tracker"][good_species][i+1] -
            microbe_trackers["good bacteria tracker"][good_species][i]
            for i in range(len(microbe_trackers["good bacteria tracker"][good_species]) - 1)
        ]

    # Calculate step-based plot data for bad bacteria
    for bad_species in microbe_trackers["bad bacteria step plot"]:
        microbe_trackers["bad bacteria step plot"][bad_species] = [
            microbe_trackers["bad bacteria tracker"][bad_species][i+1] -
            microbe_trackers["bad bacteria tracker"][bad_species][i]
            for i in range(len(microbe_trackers["bad bacteria tracker"][bad_species]) - 1)
        ]


def get_bin(location):
    # Calculate the bin index, ensuring it's clamped within [0, num_bins - 1]
    x_bin = int(min(max(location[0] // bin_size, 0), num_bins - 1))
    y_bin = int(min(max(location[1] // bin_size, 0), num_bins - 1))
    return x_bin, y_bin


def create_heatmap(dictionary, entity_type="bacteria together"):
    # Initialize an empty grid with the correct number of bins
    heatmap = np.zeros((num_bins, num_bins))

    if entity_type == "bacteria together":
        # Loop over good bacteria
        for good_species in dictionary["good object"]:
            for bacterium in dictionary["good object"][good_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1

        # Loop over bad bacteria
        for bad_species in dictionary["bad object"]:
            for bacterium in dictionary["bad object"][bad_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "good bacteria":
        for good_species in dictionary["good object"]:
            for bacterium in dictionary["good object"][good_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                # Increment count in the correct bin
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "bad bacteria":
        for bad_species in dictionary["bad object"]:
            for bacterium in dictionary["bad object"][bad_species]:
                x_bin, y_bin = get_bin(bacterium.location)
                heatmap[y_bin, x_bin] += 1

    elif entity_type == "immune_cells":
        # Loop over immune cells
        for immune_cell in dictionary['immune cells']:
            x_bin, y_bin = get_bin(immune_cell.location)
            heatmap[y_bin, x_bin] += 1

    elif entity_type == "glycogen":
        # Loop over immune cells
        for glycogen_clump in dictionary['glycogen_objects'][-1]:
            x_bin, y_bin = get_bin(glycogen_clump.location)
            heatmap[y_bin, x_bin] += 1

    elif entity_type == "ca125":
        # Loop over immune cells
        for ca in dictionary['ca125_objects'][-1]:
            x_bin, y_bin = get_bin(ca.location)
            heatmap[y_bin, x_bin] += 1

    return heatmap


def plot_heatmap(microbe_trackers, environment):

    # First figure: "Bacteria Distribution" and "Glycogen Distribution"
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 6))

    # Create each heatmap for the first figure
    bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="bacteria together")
    glycogen_heatmap = create_heatmap(environment, entity_type="glycogen")

    # Plot "Bacteria Distribution" heatmap in the first subplot
    im1 = axs1[0].imshow(bacteria_heatmap, cmap="Reds", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs1[0].set_title("Bacteria Distribution Heatmap")
    fig1.colorbar(im1, ax=axs1[0], label='Count')

    # Plot "Glycogen Distribution" heatmap in the second subplot
    im2 = axs1[1].imshow(glycogen_heatmap, cmap="Greens", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs1[1].set_title("Glycogen Distribution Heatmap")
    fig1.colorbar(im2, ax=axs1[1], label='Count')

    # Set labels for each axis in the first figure
    for ax in axs1:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

    # Adjust layout for the first figure
    plt.tight_layout()
    plt.show()

    # Second figure: "Good Bacteria Distribution" and "Bad Bacteria Distribution"
    fig2, axs2 = plt.subplots(1, 2, figsize=(12, 6))

    # Create each heatmap for the second figure
    good_bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="good bacteria")
    bad_bacteria_heatmap = create_heatmap(
        microbe_trackers, entity_type="bad bacteria")

    # Plot "Good Bacteria Distribution" heatmap in the first subplot
    im3 = axs2[0].imshow(good_bacteria_heatmap, cmap="Purples",
                         interpolation='nearest', origin='lower', extent=[0, grid_size, 0, grid_size])
    axs2[0].set_title("Good Bacteria Distribution")
    fig2.colorbar(im3, ax=axs2[0], label='Count')

    # Plot "Bad Bacteria Distribution" heatmap in the second subplot
    im4 = axs2[1].imshow(bad_bacteria_heatmap, cmap="Oranges", interpolation='nearest',
                         origin='lower', extent=[0, grid_size, 0, grid_size])
    axs2[1].set_title("Bad Bacteria Distribution")
    fig2.colorbar(im4, ax=axs2[1], label='Count')

    # Set labels for each axis in the second figure
    for ax in axs2:
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")


def plot_heatmap_seperate(heatmap, title="Bacteria Distribution", cmap="hot"):

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap=cmap, interpolation='nearest',
               origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.colorbar(label='Count')

    plt.legend()
    plt.title(title)
    plt.xlabel('X Position ')
    plt.ylabel('Y Position')
    plt.show()


def plotting(good_bacteria_proportion_tracker,label):
    

    plt.figure(figsize=(10, 10))
    plt.hist(good_bacteria_proportion_tracker, bins=10,
             density=False, label=label)
    #plt.title('Histogram of Good Bacteria for Each Trial')
    plt.xlabel('Proportion of Good Bacteria')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(fname=label, dpi=1000)
    plt.show()

def plot_together(avg_prop, end_prop):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 10))
    
    # Determine the overall range for both datasets
    combined_min = min(min(avg_prop), min(end_prop))
    combined_max = max(max(avg_prop), max(end_prop))
    bins = np.linspace(combined_min, combined_max, 11)  # 10 bins
    
    # Plot both histograms using the same bins
    plt.hist(avg_prop, bins=bins, density=False, 
             label="Average good bacteria proportion throughout the trial", 
             color='b', alpha=0.5)
    plt.hist(end_prop, bins=bins, density=False, 
             label="Proportion of good bacteria at the end of the trial", 
             color='g', alpha=0.5)
    
    plt.xlabel('Proportion of Good Bacteria')
    plt.ylabel('Counts')
    plt.legend()
    plt.savefig(fname="Distribution of trials together", dpi=1000)
    plt.show()


def my_animation(inflammation_arr, filename="inflammation_markers.gif"):
    # Initialize the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title("Inflammation Markers Over Time")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Initialize the scatter plot
    scatter = ax.scatter([], [], c='red', s=50, alpha=0.7)

    # Update function for the animation
    def update(frame):
        # Get the markers at the current time step
        current_markers = inflammation_arr[frame]

        # Extract the locations of the markers
        x_data = [marker.location[0] for marker in current_markers]
        y_data = [marker.location[1] for marker in current_markers]

        # Update the scatter plot
        scatter.set_offsets(np.c_[x_data, y_data])
        ax.set_title(f"Inflammation Markers at Time Step {frame}")

        # Return the scatter plot as a tuple
        return scatter,

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(inflammation_arr), interval=200, blit=True
    )

    ani.save(filename, writer='pillow', fps=5)

    # Display the animation
    plt.show()


# Main simulation loop refactored


def run_trial(i,good_bacteria_proportion_tracker,good_bacteria_std_tracker, test):
    print("Trial:", test + 1)

    # Read data
    data = read_data()
    estrogen_levels_raw = data[1]
    estrogen_levels = estrogen_levels_raw / max(estrogen_levels_raw)
    progesterone_levels_raw = data[2]
    progesterone_levels = progesterone_levels_raw / max(progesterone_levels_raw)

    # Initialize environment dictionary
    environment = {
        "estrogen": [],
        "progesterone": [],
        "iron": [],
        "glycogen": [],
        "glycogen_objects": [],
        "pH": [],
        "cytokine": [],
        'ca125_objects': [],
        'ca125_level': [],
        'inflammation_objects': [],
        "VEGF_level": [],
        "bacteriocins": [],
        "bacteriocins_E": []
    }
    # Initialize trackers and variables
    ca125_objects = []
    ca125_levels = []
    tumour_sizes = [0]
    inflammation_objects = []

    microbe_trackers = {
        "good bacteria tracker": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
        "bad bacteria tracker": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
        "good bacteria step": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
        "bad bacteria step": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
        "immune cells": [],
        "immune tracker": [],
        "immune production": [],
        "good bacteria step plot": {f"Good_Bacteria_{i}": [0] for i in range(1, 5)},
        "bad bacteria step plot": {f"Bad_Bacteria_{i}": [0] for i in range(1, 3)},
        "good object": {f"Good_Bacteria_{i}": [] for i in range(1, 5)},
        "bad object": {f"Bad_Bacteria_{i}": [] for i in range(1, 3)}
    }

    total_good_bacteria = []
    total_bad_bacteria = []
    total_bacteria = []
    proportion_tracker_over_time = []

    good_bacteria_count = 0
    bad_bacteria_count = 0
    glycogen_objects = []
    tumour_created = False

    for cycle in range(i):
       print(f"Cycle {cycle}")
       for step in range(DAYS):
           # Get hormone levels for the current step
           estrogen_level = estrogen_levels[step]
           progesterone_level = progesterone_levels[step]
           pH_levels = environment["pH"]
           # bacteriocins secreted at each step
           bacteriocin_objects = []
           bacteriocin_objects_E = []

           # Create a tumor at a specific cycle and step if conditions are met
           if not tumour_created and cycle == CYCLE_START and step == STEP_START:
               tumour = create_tumour(cycle, step)
               tumour_created = True  # Set the flag to indicate tumor is now present

           # Call tumour_interaction only if the tumor is present
           if tumour_created:
               tumour_interaction(
                   tumour, ca125_objects, tumour_sizes, environment["VEGF_level"])
           else:
               tumour_sizes.append(0)

           # Initialize the state for the current step, passing existing glycogen_objects
           current_state = initialize_state(
               estrogen_level, progesterone_level, good_bacteria_count,
               bad_bacteria_count, glycogen_objects, ca125_objects,
               inflammation_objects, tumour if tumour_created else None,
               step, pH_levels
           )

           # Store hormone levels and glycogen state in the environment
           environment["estrogen"].append(
               current_state['estrogen_level'])
           environment["progesterone"].append(
               current_state['progesterone_level'])
           environment["iron"].append(current_state['iron_level'])
           environment["glycogen"].append(
               current_state['glycogen_level'])
           environment["glycogen_objects"].append(
               [glycogen for glycogen in current_state['glycogen_objects']]
           )
           environment["cytokine"].append(
               current_state['cytokine_level'])
           environment["pH"].append(current_state['pH'])
           environment["ca125_level"].append(
               current_state["ca125_levels"])

           environment["ca125_objects"].append(
               [CA for CA in current_state["ca125_objects"]])

           environment["inflammation_objects"].append(
               [marker for marker in current_state['inflammation_objects']] if current_state['inflammation_objects'] else [])
           environment["VEGF_level"].append(
               current_state["VEGF_level"])

           # Handle microbe interactions
           handle_microbes(
               current_state, microbe_trackers, glycogen_objects, bacteriocin_objects, bacteriocin_objects_E)
           environment["bacteriocins"].append(bacteriocin_objects)
           environment["bacteriocins_E"].append(bacteriocin_objects_E)

           # Generate new immune cells
           generate_immune_cells(current_state, microbe_trackers)

           # Handle immune cell interactions
           handle_immune_cells(microbe_trackers)

           # Update step trackers
           update_trackers(microbe_trackers)

           # Only interact with CA125 if a tumor is present
           if tumour_created:
               ca125_interact(
                   ca125_objects, microbe_trackers['good object'], microbe_trackers['immune cells'])

           handle_inflamation_markers(
               environment['inflammation_objects'][-1], microbe_trackers['immune cells'])
           # Update chemical levels in the state'''

           update_chemical_levels(
               environment, current_state, good_bacteria_count, bad_bacteria_count, step)

           # Calculate total good, bad, and overall bacteria counts
           good_bacteria_count = sum(
               microbe_trackers["good bacteria tracker"][species][-1]
               for species in microbe_trackers["good bacteria tracker"]
           )
           bad_bacteria_count = sum(
               microbe_trackers["bad bacteria tracker"][species][-1]
               for species in microbe_trackers["bad bacteria tracker"]
           )
           total_bacteria_count = good_bacteria_count + bad_bacteria_count

           # Store total bacteria counts
           total_good_bacteria.append(good_bacteria_count)
           total_bad_bacteria.append(bad_bacteria_count)
           total_bacteria.append(total_bacteria_count)

           # Track the proportion of good bacteria over time
           current_good_proportion = good_bacteria_count / \
               total_bacteria_count if total_bacteria_count > 0 else 0
           proportion_tracker_over_time.append(
               current_good_proportion)

    avg_prop = np.mean(proportion_tracker_over_time)
    std_prop = np.std(proportion_tracker_over_time)

   # Track good bacteria proportions at the end of the trial
    good_bacteria_proportion_end = good_bacteria_count / \
       (good_bacteria_count + bad_bacteria_count) if (good_bacteria_count +
                                                      bad_bacteria_count) > 0 else 0

    good_bacteria_proportion_tracker.append(avg_prop)
    good_bacteria_std_tracker.append(std_prop)

   # Add total bacteria to the microbe trackers
    microbe_trackers["total good"] = total_good_bacteria
    microbe_trackers["total bad"] = total_bad_bacteria
    microbe_trackers["total bacteria"] = total_bacteria
    microbe_trackers['proportion'] = proportion_tracker_over_time

    return avg_prop, std_prop,good_bacteria_proportion_end

def simulation_loop():
    std_tracker = []
    average_tracker = []
    end_prop_tracker = []

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in cycle_lengths:
            print("Test of cycle length", i)
            good_bacteria_proportion_tracker = []
            good_bacteria_std_tracker = []
            for test in range(TRIALS):
                futures.append(executor.submit(run_trial, i,good_bacteria_proportion_tracker,good_bacteria_std_tracker, test))

        for future in futures:
            avg_prop, std_prop,end_prop = future.result()
            average_tracker.append(avg_prop)
            std_tracker.append(std_prop)
            end_prop_tracker.append(end_prop)

    return average_tracker, std_tracker,end_prop_tracker

def main():
    average_tracker, std_tracker,end_prop_tracker = simulation_loop()
    plotting(average_tracker,"Proportion averaged over all cycles")
    plotting(end_prop_tracker,"Proportion at end of trial")
    plot_together(average_tracker, end_prop_tracker)
    weights_arr = []
    for i in std_tracker:
        weight = 1/(i)**2
        weights_arr.append(weight)

    weighted_avg = np.average(
        average_tracker, weights=weights_arr)
    weight_sum = np.sum(weights_arr)
    weighted_std = np.sqrt(1/weight_sum)
    
    print(
        f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {weighted_avg:.2f} using a cumulative avg over time")
    print(
        f"Average Proportion of good bacteria over {TRIALS} TRIALS is: {np.mean(end_prop_tracker):.2f} using the end values")
    # Calculate and print the standard deviation of the proportion of good bacteria across all trials
   
    print(
        f"The standard deviation of the trials is: {weighted_std:.2f} using the cumulative average")
    print(
        f"The standard deviation of the trials is: {np.std(end_prop_tracker):.2f} using the end values")
    return average_tracker,std_tracker
    # Further processing and plotting


if __name__ == "__main__":
    # Run the simulation only if this script is executed directly
    start_time = time.time()
    average_tracker,std_tracker = main()
    print("--- %s seconds ---" % (time.time() - start_time))
    

    