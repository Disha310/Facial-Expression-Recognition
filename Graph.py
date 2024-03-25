import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

timestamps = []
emotions = []

def update_graph(frame, timestamps, emotions):
    ax.clear()
    ax.plot(timestamps[:frame+1], emotions[:frame+1], marker='o')
    ax.set_title('Facial Expression Detection Over Time')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Emotion')

if __name__ == "__main__":
    plt.style.use('default')
    fig, ax = plt.subplots()

    with open('database_information.txt', 'r') as file:
        for line in file:
            data = line.strip().split(' - ')
            timestamps.append(data[1])
            emotions.append(data[0])

    ani = FuncAnimation(fig, update_graph, frames=len(timestamps), fargs=(timestamps, emotions), interval=1000)
    plt.show()