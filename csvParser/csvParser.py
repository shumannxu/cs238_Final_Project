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
        Quarter,Time,Down,ToGo,Location,home,away,Detail,EPB,EPA = self.file.readline().strip().split(',')
        self.homeTeam = home
        self.awayTeam = away
        self.previousState = None

    def readLine(self):
        self.file.readline()
        for row in self.file:
            if row == '\n':
                break
            Quarter,Time,Down,ToGo,Location,home,away,Detail,EPB,EPA = row.split(',')
            endzoneDistance = None
            self.previousState = None
            self.checkCurrPossession(Detail, Location)
            # Handles edge cases for nontarget team possessing the ball, extra point, and plays that don't have downs
            if self.currPossession != self.targetTeam or 'point' in Detail or not Down:
                continue
            # If the target team who possesses the ball is on their own turf, then the endzone distance is 100 - the yard line
            if self.currPossession == Location.split(' ')[0]:
                endzoneDistance = 100 - int(Location.split(' ')[1])
            else:
                endzoneDistance = int(Location.split(' ')[1])
            currentAction = self.createAction(Detail)
            # If there exists a previous state, then we can link the previous state to the current state
            if self.data and self.data[-1][0]:
                self.previousState = self.data[-1]
            #s, a, r, s' where s = (down, toGo, endzoneDistance), a = action, r = reward, s' = next state or 'TERMINAL'
            newState = [self.createState(Down, ToGo, endzoneDistance) , currentAction, self.createReward(EPB, EPA), 'TERMINAL' if self.isTerminalState(currentAction, Detail) else '']
            if self.previousState and not 'TERMINAL' in self.previousState[3]:
                self.previousState[3] = newState[0]
            self.data.append(newState)



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
    
    def checkCurrPossession(self, detail, Location):
        if 'kicks off' in detail:
            self.currPossession = Location.split(' ')[0]
        elif 'punts' in detail or 'returned' in detail: # not sure if returned applies for fumbles all the time (akwon check)
            if self.currPossession == self.homeTeam:
                self.currPossession = self.awayTeam
            else:
                self.currPossession = self.homeTeam

    def isTerminalState(self, action, Detail):
        if action == actions.fieldGoal or action == actions.punt or 'touchdown' in Detail or 'intercept' in Detail or ('fumble' in Detail and 'recovered' in Detail) or 'turnover on downs' in Detail:
            return True
        else:
            return False
        
example = csvParser('data/sfVsJaxWeek10.csv', 'SFO')
example.readLine()
print(example.data)
