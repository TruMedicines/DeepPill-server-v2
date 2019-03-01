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

    def _convert_to_degress(self, value):
        """
        Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
        :param value:
        :type value: exifread.utils.Ratio
        :rtype: float
        """
        d = float(value.values[0].num) / float(value.values[0].den)
        m = float(value.values[1].num) / float(value.values[1].den)
        s = float(value.values[2].num) / float(value.values[2].den)

        return d + (m / 60.0) + (s / 3600.0)

    def addPill(self, imageData):
        buffer = io.BytesIO()
        buffer.write(imageData)
        buffer.seek(0)

        tags = exifread.process_file(buffer)


        if 'GPS GPSLatitude' in tags:
            gps_latitude = tags['GPS GPSLatitude']
            gps_longitude = tags['GPS GPSLongitude']
            gps_latitude_ref = tags['GPS GPSLatitudeRef']
            gps_longitude_ref = tags['GPS GPSLongitudeRef']

            lat = self._convert_to_degress(gps_latitude)
            if gps_latitude_ref.values[0] != 'N':
                lat = 0 - lat

            lon = self._convert_to_degress(gps_longitude)
            if gps_longitude_ref.values[0] != 'E':
                lon = 0 - lon

            self.pills.append({"lat": lat, "lng": lon})
            self.writeDB()


    def getPills(self):
        return self.pills