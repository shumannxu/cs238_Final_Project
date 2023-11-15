from datetime import datetime, timedelta
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
    def __init__(self, data, drives, targetTeam):
        # every key in data is the following 
        self.data = []
        self.file = open(data, 'r')
        self.drives = open(drives, 'r')
        self.possessions = {key: [] for key in range(1,5)}
        self.readDrives()
        self.targetTeam = targetTeam
        self.currPossession = None
        Quarter,Time,Down,ToGo,Location,home,away,Detail,EPB,EPA = self.file.readline().strip().split(',')
        self.homeTeam = home
        self.awayTeam = away
        self.previousState = None

    def readDrives(self):
        self.drives.readline()
        for row in self.drives:
            if row == '\n':
                break
            numDrive, Quarter,Time,Location,Plays,Length,NetYds,Result = row.split(',')
            duration = (self.convertTime(Time), self.subtractDuration(Time, Length))
            self.possessions[int(Quarter)].append(duration)

    def readLine(self):
        self.file.readline()
        for row in self.file:
            if row == '\n':
                break
            Quarter,Time,Down,ToGo,Location,home,away,Detail,EPB,EPA = row.split(',')
            endzoneDistance = None
            self.previousState = None
            self.currPossession = self.isTargetPossession(Quarter, Time)
            # Handles edge cases for nontarget team possessing the ball, extra point, and plays that don't have downs
            if not self.currPossession or 'point' in Detail or not Down:
                continue
            # If the target team who possesses the ball is on their own turf, then the endzone distance is 100 - the yard line
            if self.targetTeam == Location.split(' ')[0]:
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

    def convertTime(self, time):
        time_format = '%M:%S'
        return datetime.strptime(time, time_format)
    
    def subtractDuration(self, time_str, duration_str):
        # Parse time and duration strings
        time_format = '%M:%S'
        time = datetime.strptime(time_str, time_format)

        duration = datetime.strptime(duration_str, time_format)

        # Subtract duration from time
        result = time - duration
        
        # Handle negative result by wrapping around to the next quarter
        if result < timedelta():
            result += timedelta(minutes=15)

        # Format the result as minutes and seconds
        result_minutes, result_seconds = divmod(result.seconds, 60)
        
        # Convert back to 'MM:SS' string format
        new_time = '{:02d}:{:02d}'.format(result_minutes, result_seconds)
        return self.convertTime(new_time)



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
    
    def isTargetPossession(self, quarter, time):
        time = self.convertTime(time)
        checkQuarter = self.possessions[int(quarter)]
        for possession in checkQuarter:
            if possession[0] >= time and time >= possession[1]:
                return True
        return False
        

    def isTerminalState(self, action, Detail):
        if action == actions.fieldGoal or action == actions.punt or 'touchdown' in Detail or 'intercept' in Detail or ('fumble' in Detail and 'recovered' in Detail) or 'turnover on downs' in Detail:
            return True
        else:
            return False
        
example = csvParser('data/sfVsJaxWeek10.csv', 'data/sfDrivesWeek10.csv', 'SFO')
example.readLine()
print(example.data)
