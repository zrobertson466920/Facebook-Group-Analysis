import matplotlib.pyplot as plt
import fgroupanalysis as fg
from fgroupanalysis import Timeline

def main():

    # Open and Read File
    timeline = Timeline(raw = 'message.json')

    # Splits the Messages By Sender
    timelines = timeline.partition()

    # Flatten Structure into List of Messages
    text = timeline.raw_text()

    times = fg.activity_plot(timeline.get_timestamps(),0)

    plt.figure()
    plt.hist(times,bins = 24, density = True)
    plt.xlabel("Time (24hr)")
    plt.ylabel("Percent of Messages")
    plt.title("Activity Plot of Group")
    plt.show()

main()
