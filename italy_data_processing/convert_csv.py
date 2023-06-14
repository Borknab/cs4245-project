#
# The purpose of this script is to preprocess csv file taken from
# http://dati.istat.it/Index.aspx?QueryId=37850&lang=en#
# 
# The 'it_data.csv' file contains the soy bean yields per each Italian province from 2010 to 2015 (other dates can be chosen on the website).
# The following script maps each province's name to a GAUL code, converts the yield units to bushels per acre, and saves the results in 'it_yield_data.csv' file. 
# 

import csv

csvfile = open('it_data.csv', 'r')
reader = csv.reader(csvfile, delimiter=',')
it_data = {}

mapping = [ [ 18400, "Firenze" ], [ 18407, "Prato" ], [ 18323, "Catanzaro" ], [ 18325, "Crotone" ], [ 18327, "Vibo Valentia" ], [ 18336, "Forli'" ], [ 18342, "Rimini" ], [ 18317, "Chieti" ], [ 18318, "L'Aquila" ], [ 18319, "Pescara" ], [ 18320, "Teramo" ], [ 18321, "Matera" ], [ 18322, "Potenza" ], [ 18324, "Cosenza" ], [ 18326, "Reggio Calabria" ], [ 18328, "Avellino" ], [ 18329, "Benevento" ], [ 18330, "Caserta" ], [ 18332, "Napoli" ], [ 18333, "Salerno" ], [ 18334, "Bologna" ], [ 18337, "Modena" ], [ 18338, "Parma" ], [ 18339, "Piacenza" ], [ 18340, "Ravenna" ], [ 18341, "Reggio Emilia" ], [ 18347, "Frosinone" ], [ 18348, "Latina" ], [ 18349, "Rieti" ], [ 18350, "Roma" ], [ 18351, "Viterbo" ], [ 18352, "Genova" ], [ 18353, "Imperia" ], [ 18354, "La Spezia" ], [ 18355, "Savona" ], [ 18367, "Ancona" ], [ 18368, "Ascoli Piceno" ], [ 18369, "Macerata" ], [ 18370, "Pesaro" ], [ 18371, "Campobasso" ], [ 18372, "Isernia" ], [ 18373, "Alessandria" ], [ 18374, "Asti" ], [ 18376, "Cuneo" ], [ 18378, "Torino" ], [ 18381, "Bari" ], [ 18382, "Brindisi" ], [ 18383, "Foggia" ], [ 18384, "Lecce" ], [ 18385, "Taranto" ], [ 18386, "Cagliari" ], [ 18387, "Nuoro" ], [ 18388, "Oristano" ], [ 18389, "Sassari" ], [ 18390, "Agrigento" ], [ 18391, "Caltanissetta" ], [ 18392, "Catania" ], [ 18393, "Enna" ], [ 18394, "Messina" ], [ 18395, "Palermo" ], [ 18396, "Ragusa" ], [ 18397, "Siracusa" ], [ 18398, "Trapani" ], [ 18399, "Arezzo" ], [ 18401, "Grosseto" ], [ 18402, "Livorno" ], [ 18403, "Lucca" ], [ 18404, "Massa-carrara" ], [ 18405, "Pisa" ], [ 18406, "Pistoia" ], [ 18408, "Siena" ], [ 18411, "Perugia" ], [ 18412, "Terni" ], [ 18335, "Ferrara" ], [ 18343, "Gorizia" ], [ 18344, "Pordenone" ], [ 18345, "Trieste" ], [ 18346, "Udine" ], [ 18356, "Bergamo" ], [ 18357, "Brescia" ], [ 18358, "Como" ], [ 18359, "Cremona" ], [ 18360, "Lecco" ], [ 18361, "Lodi" ], [ 18362, "Mantova" ], [ 18363, "Milano" ], [ 18364, "Pavia" ], [ 18365, "Sondrio" ], [ 18366, "Varese" ], [ 18375, "Biella" ], [ 18377, "Novara" ], [ 18379, "Verbania" ], [ 18380, "Vercelli" ], [ 18409, "Bolzano" ], [ 18410, "Trento" ], [ 18413, "Aosta" ], [ 18414, "Belluno" ], [ 18415, "Padova" ], [ 18416, "Rovigo" ], [ 18417, "Treviso" ], [ 18418, "Venezia" ], [ 18419, "Verona" ], [ 18420, "Vicenza" ] ]
mapping = {value: key for (key, value) in mapping}

for row in reader:
    if row[1] not in it_data: it_data[row[1]] = {}
    if row[-5] not in it_data[row[1]]: it_data[row[1]][row[-5]] = {}
    if row[3] == "harvested production - quintals ": it_data[row[1]][row[-5]]['value'] = int(row[-3])
    if row[3] == "total area - hectares": it_data[row[1]][row[-5]]['area'] = int(row[-3])

it_data.pop('Territory')
regionss = set()

with open('it_yield_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Province', 'Year', 'Value'])
    for region in it_data:
        for year in it_data[region]:
            year_data = it_data[region][year]
            if 'area' in year_data and 'value' in year_data and region in mapping:
                quintals = year_data['value']
                kg = quintals * 100
                kg_per_hectare = kg / year_data['area']
                bushes_per_acre = kg_per_hectare * 0.01487
                # write bushes_per_acre with 1 decimal place
                writer.writerow([mapping[region], year, round(bushes_per_acre, 1)])

csvfile.close()