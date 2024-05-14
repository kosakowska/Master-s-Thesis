import fileinput
import sys
import argparse
import os
import statistics
import math

# Class to manage distance measurements
class MeasureDistance:
    def __init__(self, distanceTemp):
        self.distance = str(distanceTemp)  # Initialize with a distance value converted to string

    filename = ""  # Filename for storing results
    command = ""  # Command to perform the measurement

    # Generate a filename based on the distance
    def createFilename(self):
        self.filename = "pomiar" + self.distance + ".txt"
        return self.filename

    # Create a command to be executed for measuring
    def createCommand(self):
        self.command = "iw wlp1s0 measurement ftm_request /home/gosia/conf |head -3| tail -1 >> " + self.filename
        return self.command

    # Static method to execute the command 100 times
    @staticmethod
    def measureDistance(commandT):
        for i in range(100):
            os.system(commandT)

# Class to handle calculation of distances from file data
class CalculateDistance:
    listOfDistances = list()  # A list to store distances

    # Extract distances from a file and store them in a list
    def extractDistanceFromFileToList(self, filename):
        with open(filename, "r") as file:
            for line in file:
                table = line.split()
                for index, word in enumerate(line.split()):
                    if word == "distance:":
                        if int(table[index + 1]) != 0:
                            self.listOfDistances.append(int(table[index + 1]))
                    elif "=" in word:
                        self.listOfDistances.append(word)

    # Write calculated distances to a file
    def writeDistancesToFile(self, paramName, distance):
        tempList = list()
        bwCounter = 1
        filenameForDistances = paramName + distance + ".csv"
        os.system("echo " + paramName + " >> " + filenameForDistances)
        for value in self.listOfDistances:
            if "=" not in str(value) and value != "end":
                tempList.append(value)
            elif tempList:
                for measurement in tempList:
                    calculatedDistance = 43.26 * math.exp(0.0023 * measurement)
                    os.system("echo " + str(bwCounter) + "," + str(calculatedDistance) + "," + " >> " + filenameForDistances)
                tempList.clear()
                bwCounter += 1

# Class to change and manipulate parameters
class ChangeParameter:
    def __init__(self, commandTemp, filenameTemp):
        self.command = str(commandTemp)
        self.filename = str(filenameTemp)

    paramName = ""
    paramNameWithValue = ""
    allowedParams = ["spb", "asap", "bursts_exp", "burst_period", "retries", "burst_duration", "lci", "civic"]
    spbValues = list(range(1, 32))
    burstExpValues = list(range(0, 16))
    burstPeriodValues = list(range(0, 32))
    retriesValues = list(range(0, 4))
    burstDurationValues = list(range(0, 16))

    # Method to replace or add parameters in a configuration file
    def replaceOrAddParameter(self):
        self.resetConfFile()
        wordToChange = ""
        defaultLine = "00:c2:c6:e5:28:9c bw=20 cf=2412 asap"
        with fileinput.FileInput("/home/gosia/conf", inplace=True, backup='.bak') as file:
            for line in file:
                for index, word in enumerate(line.split()):
                    if self.paramName == word:
                        wordToChange = word
                if wordToChange:
                    print(line.replace(wordToChange, self.paramName), end='')
                else:
                    lineToChange = defaultLine + " " + self.paramNameWithValue
                    print(lineToChange, end='')

    # Reset the configuration file to default settings
    @staticmethod
    def resetConfFile():
        defaultLine = "00:c2:c6:e5:28:9c bw=20 cf=2412 asap"
        with fileinput.FileInput("/home/gosia/conf", inplace=True, backup='.bak') as file:
            for line in file:
                print(defaultLine)

    # Check if the parameter name is allowed
    def checkIfAllowedArgument(self, paramName):
        if paramName not in self.allowedParams:
            raise ValueError("Wrong argument passed!\nPossible arguments: spb, asap, bursts_exp, burst_period, "
                             "retries, burst_duration, lci, civic")

    # Combine parameter name with its value for configuration
    def concatenateNameWithValue(self, paramName, value):
        self.paramNameWithValue = paramName + "=" + str(value)

    # Iterate through all values of a parameter, updating configuration and measuring distance
    def goThroughAllValues(self, paramName, values):
        for value in values:
            self.concatenateNameWithValue(paramName, value)
            self.replaceOrAddParameter()
            self.command = "echo " + self.paramNameWithValue + " >> " + self.filename
            os.system(self.command)
            MeasureDistance.measureDistance(self.command)
            print(self.paramNameWithValue)
        os.system("echo end >> " + self.filename)

    # Process all values of a specific parameter
    def goThroughAllValuesOfParam(self, paramName):
        if paramName in ["spb", "bursts_exp", "burst_period", "retries", "burst_duration"]:
            self.goThroughAllValues(paramName, getattr(self, paramName + 'Values'))
        elif paramName in ["asap", "lci", "civic"]:
            self.paramNameWithValue = paramName

# Main entry point for script execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", choices=ChangeParameter.allowedParams, help='Special testing value')
    distance = input("Enter distance: ")
    md = MeasureDistance(distance)
    filename = md.createFilename()
    command = md.createCommand()
    cp = ChangeParameter(command, filename)
    cp.paramName = sys.argv[2]
    cp.checkIfAllowedArgument(cp.paramName)
    cp.goThroughAllValuesOfParam(cp.paramName)
    cl = CalculateDistance()
    cl.extractDistanceFromFileToList(md.filename)
    cl.writeDistancesToFile(cp.paramName, distance)
    os.system("rm " + filename)
