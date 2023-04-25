import random
import csv


def write_to_tsv(filename, data):
    with open(filename, mode='w', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        for row in data:
            tsv_writer.writerow(row)
            
market_attitudes = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

class Startup:
    def __init__(self, sector, risk):
        self.sector = sector
        self.risk = risk

    def __str__(self):
        return f'Sector: {self.sector}\nRisk: {self.risk:.2f}\n'

def generate_startups(num_startups):
    sectors = [1, 2, 3, 4, 5]
    startups = []
    for _ in range(num_startups):
        sector = random.choice(sectors)
        risk = random.random()
        startup = Startup(sector, risk)
        startups.append(startup)
    return startups

def simulate(startup, investment):
    if (startup.risk > random.random()*1.2): #fail from risk
        return 0
    if (market_attitudes[startup.sector] < 0): #fail from bad sector
        return 0
    if (investment == False and startup.risk > random.random()*1.2): #fail from no investment
        return 0
    return 1


def main():
    num_startups = 5000
    startups = generate_startups(num_startups)
    results = []
    for startup in startups:
        results.append([startup.sector, round(startup.risk, 3), simulate(startup, True)])
    write_to_tsv("data.dat", results)
        

if __name__ == '__main__':
    main()
