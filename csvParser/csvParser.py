class downs(enumerate):
    firstDown = '1'
    secondDown = '2'
    thirdDown = '3'
    fourthDown = '4'

class actions(enumerate):
    passing = 1
    run = 2
    fieldGoal = 3
    punt = 4

class terminalStates(enumerate):
    touchdown = 1
    fieldGoal = 2
    punt = 3
    turnover = 4 # turnover on downs, interception, fumble, and missed field goal

class csvParser(object):
    def __init__(self, data, targetTeam):
        # every key in data is the following 
        self.data = []
        self.file = open(data, 'r')
        self.targetTeam = targetTeam
        self.currPossession = None

    def readLine(self):
        self.file.readline()
        self.file.readline()
        for row in self.file:
            if row == '\n':
                break
            Quarter,Time,Down,ToGo,Location,SFO,away,Detail,EPB,EPA = row.split(',')
            endzoneDistance = None
            if 'kicks off' in Detail:
                self.currPossession = Location.split(' ')[0]
            if self.currPossession != self.targetTeam:
                continue
            # If the target team who possesses the ball is on their own turf, then the endzone distance is 100 - the yard line
            if self.currPossession == Location.split(' ')[0]:
                endzoneDistance = 100 - int(Location.split(' ')[1])
            else:
                endzoneDistance = int(Location.split(' ')[1])
            currentAction = self.createAction(Detail)
            # Appending every state, action, reward to a list; however, we need to link to a future state so we need to be able to handle terminal states and transition states
            self.data.append((self.createState(Down, ToGo, endzoneDistance, currentAction, self.isTerminalState(currentAction, Detail)), self.createReward(EPB, EPA), None))
        self.linkStates()

    def linkStates(self):
        # Linking states to future states, this might be buggy/tricky since we need to handle terminal states
        for i in range(len(self.data)):
            if i == len(self.data) - 1:
                self.data[i][2][4] = None
            else:
                self.data[i][2][4] = self.data[i+1][0]

    def createState(self, down, toGo, endzoneDistance):
        return (down, toGo, endzoneDistance)
    
    def createAction(self, detail):
        if 'pass' in detail:
            return actions.passing
        elif 'run' in detail:
            return actions.run
        elif 'field goal' in detail:
            return actions.fieldGoal
        elif 'punt' in detail:
            return actions.punt
        else:
            return None
    
    def createReward(self, EPB, EPA):
        return float(EPA) - float(EPB)

    def isTerminalState(self, action, Detail):
        if action == actions.fieldGoal or action == actions.punt or 'touchdown' in Detail or 'intercept' in Detail or ('fumble' in Detail and 'recovered' in Detail) or 'turnover on downs' in Detail:
            return True
        else:
            return False
        
example = csvParser('data/sfVsJaxWeek10.csv', 'SFO')
example.readLine()
