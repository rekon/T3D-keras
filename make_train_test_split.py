from random import shuffle
import csv
import glob

action_classes = ['HorseRace', 'YoYo', 'SoccerJuggling', 'MilitaryParade', 'BaseballPitch', 'JugglingBalls', 'BenchPress', 'Biking', 'VolleyballSpiking', 'PlayingGuitar', 'ThrowDiscus', 'SalsaSpin', 'PlayingPiano', 'PoleVault', 'Mixing', 'PlayingViolin', 'CleanAndJerk', 'Basketball', 'HulaHoop', 'JumpingJack', 'RopeClimbing', 'GolfSwing', 'PizzaTossing', 'Fencing', 'TrampolineJumping', 'Billiards', 'Nunchucks', 'PommelHorse', 'SkateBoarding', 'HorseRiding', 'TennisSwing', 'TaiChi', 'Diving', 'Drumming', 'WalkingWithDog', 'Skijet', 'Lunges', 'Rowing', 'PlayingTabla', 'Punch', 'BreastStroke', 'RockClimbingIndoor', 'JavelinThrow', 'PushUps', 'Kayaking', 'Skiing', 'Swing', 'PullUps', 'JumpRope', 'HighJump']
print(len(set(action_classes)))
def create_csvs():
    train = []
    test = []
    total_train = []
    total_test = []

    for myclass, directory in enumerate(action_classes):
        for filename in glob.glob('UCF50/{}/*.avi'.format(directory)):
            group = ((filename.split('/')[-1]).split('.')[0]).split('_')[-2]

            if group in ['g01','g02','g02','g04', 'g05']:
                test.append([filename, myclass, directory])
            else:
                train.append([filename, myclass, directory])

    shuffle(train)
    shuffle(test)
    # print('train', len(total_train))
    # print('test', len(total_test))

    with open('train.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'sport'])
        mywriter.writerows(train)
        print('Training CSV file created successfully')

    with open('test.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile)
        mywriter.writerow(['path', 'class', 'sport'])
        mywriter.writerows(test)
        print('Testing CSV file created successfully')

    print('CSV files created successfully')


if __name__ == "__main__":
    create_csvs()
