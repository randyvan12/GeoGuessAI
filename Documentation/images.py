from geopy.geocoders import Nominatim
import os
import pandas as pd
import shutil
geolocator = Nominatim(user_agent="MyApp")

set = set()
data = pd.read_csv("D:\\archive\\picture_coords.csv")
i = 0
for coordinate in data.iterrows():
    latitude = coordinate[1][0]
    longitude = coordinate[1][1]
    coordinates = str(latitude) + ", " + str(longitude)
    location = geolocator.reverse(coordinates)
    address = location.raw['address']
    city = address.get('city', '')
    newpath = 'D:\\archive\\' + city
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    set.add(city)
    image = 'street_view_' + str(i) + '.jpg'
    try:
        shutil.move('D:/archive/' + image, 'D:/archive/' + city + '/' + image)
    except:
        print("file not found")
    
    print(i)
    i += 1 
print('done')
#print(data.at[0, 'Latitude'])
# coordinates = "40.8644125, -73.8932922"
# location = geolocator.reverse(coordinates)

# print(location.address)
# address = location.raw['address']

# # Traverse the data
# city = address.get('city', '')
# print(city)

# newpath = 'D:\\archive\\' + city
# if not os.path.exists(newpath):
#     os.makedirs(newpath)