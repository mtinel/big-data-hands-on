import json
import time
#import urllib
import urllib.request

from kafka import KafkaProducer

#API_KEY = "XXX" # FIXME Set your own API key here
API_KEY = "a44bff4661b1435a625ea7b661a3f6410386bcf3"
BOOTSTRAP_SERVERS = 'kafka:9092'

url = "https://api.jcdecaux.com/vls/v1/stations?apiKey={}".format(API_KEY)
producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS)
stations = {}

while True:
    response = urllib.request.urlopen(url)
    stations_upd8 = json.loads(response.read().decode())
    for station in stations_upd8:
        station_number = station["number"]
        contract = station["contract_name"]
        available_bikes = station["available_bikes"]

        if contract not in stations:
            stations[contract] = {}

        city_stations = stations[contract]

        if station_number not in city_stations:
            city_stations[station_number] = available_bikes
        else:
            count_diff = available_bikes - city_stations[station_number]
            empty_shift = (available_bikes == 0  or city_stations[station_number] == 0) and count_diff != 0

            if empty_shift:
                city_stations[station_number] = available_bikes
                producer.send("empty-stations", json.dumps(station).encode(),
                        key=contract.encode()) # dispatch each city always on same partition
                print("{} - {} : {} bikes available".format(station["number"], station["contract_name"], station["available_bikes"]))
    print("-------------")
#    print("{} Produced {} station records".format(time.time(), len(stations)))
    time.sleep(1)