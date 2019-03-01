import pickle
import exifread
import io
import os.path

class PillDB:
    """This class is a very very quick, rudimentary system for storing the GPS coordinates of pills that have been uploaded to the system.

    It is not in anyway meant to act like a full database."""
    def __init__(self):
        if os.path.exists('pilldb.bin'):
            self.readDB()
        else:
            self.pills = []

    def writeDB(self):
        with open('pilldb.bin', 'wb') as file:
            pickle.dump(self.pills, file)

    def readDB(self):
        with open('pilldb.bin', 'rb') as file:
            self.pills = pickle.load(file)

    def addPill(self, imageData):
        buffer = io.BytesIO()
        buffer.write(imageData)

        tags = exifread.process_file(buffer)

        if 'GPS GPSLatitude' in tags:
            gps_latitude = tags['GPS GPSLatitude']
            gps_longitude = tags['GPS GPSLongitude']

            self.pills.append({"lat": gps_latitude, "lng": gps_longitude})
            self.writeDB()


    def getPills(self):
        return self.pills