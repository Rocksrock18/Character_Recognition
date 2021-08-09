import csv

class DataWriter():
    def __init__(self):
        pass

    def get_header(self):
            return ["Generation", "Highest Accuracy", "Average Accuracy", "Highest Score", "Average Score", "Average Network Size"]

    def init_table(self, file):
        with open(file, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(self.get_header())


    def write_row(self, file, row):
        with open(file, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(row)
