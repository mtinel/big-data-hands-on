import json
from kafka import KafkaConsumer

BOOTSTRAP_SERVERS = 'kafka:9092'

consumer = KafkaConsumer("empty-stations", bootstrap_servers=BOOTSTRAP_SERVERS, group_id="velib-monitor-stations")
cities = {}

for message in consumer:
    station = json.loads(message.value.decode())
    contract = station["contract_name"]
    available_bikes = station["available_bikes"]

    if contract not in cities:
        cities[contract] = 0

    if available_bikes == 0:
        cities[contract] += 1
        print("no more bikes at {} ({}). {} empty station(s) in this city".format(
            station["address"], contract, cities[contract]))
#    elif cities[contract] != 0: # We don't know how many stations are empty at startup
    else:
        cities[contract] -= 1
