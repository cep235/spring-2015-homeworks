from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np

''''''''''''''''''''''''''''''''''''
'        F U N C T I O N S         '
''''''''''''''''''''''''''''''''''''

def fixLocations(food_inspections):

	'''
	This function attempts to recover missing geolocations in the dataset.

	INPUT:
		@param food_inspections: Food inspections Dataframe with missing geolocations

	RETURNS:
		@return food_inspections: The original DataFrame, but with the missing geolocation entries now filled in
	'''

	#If there is no geolocation for a restaurant/establishment, we want to get it. We can use geopy
	geolocator = Nominatim()
	ADDRESSES = {} #initialize dictionary which will keep track of all the unique addresses
	for idx, row in food_inspections.iterrows():
		print idx

		#If the row is null, let's see if we can get a geolocation!
		if (pd.isnull(row['Location'])):

			#Form an address to be put into the algorithm
			try:
				address = row['Address'] + " " + row['City'] + " " + "Massachusetts"

			except: #This happens when 'Address', 'City', etc. are NaN
				ADDRESSES[address] = None
				continue

			#Check if the current address is already in the dictionary of addresses (we don't want to look up the same address twice)
			if (address not in ADDRESSES):

				try:
					location = geolocator.geocode(address, timeout=1) #try to get location, timeout after 10 sec
					loc = (location.latitude, location.longitude)

				#Sometimes we have addresses like '466 CENTRE'. There is no 'ST', 'RD', etc. in the name... So
				#we need to figure out the addresss here...

				except: #Formerly 'except AttributeError'

					temp = {}
					roads = ['ST', 'RD', 'DR', 'AVE', 'CTR', 'BLVD', 'LN', 'WAY', 'PL', 'TERRACE']

					for road in roads:
						address = row['Address'] + " " + road + " " + row['City'] + " " + row['State']

						try:
							location = geolocator.geocode(address, timeout=1) 
							loc = (location.latitude, location.longitude)
						except: #except all
							continue
						else:
							break;

					ADDRESSES[address] = loc #add address to ADDRESSES dictionary
					
					#continue

			else:
				loc = ADDRESSES[address] #address is already in dictionary, so extract it

			food_inspections.loc[idx,'Location'] = str(loc)

	return food_inspections



''''''''''''''''''''''''''''''''''''
'      M A I N    S C R I P T      '
''''''''''''''''''''''''''''''''''''

#Load data
original_dataset = pd.read_csv('Food_Establishment_Inspections.csv', sep=',',low_memory=False)

#Get list of original locations & find how many are missing
original_locs = original_dataset['Location']
original_NaN_inds = pd.isnull(original_locs).nonzero()[0]

#Fill in missing locs
fixed_dataset = fixLocations(original_dataset)

#Get list of newly filled-in locations & find how many are missing
fixed_locs = fixed_dataset['Location']
fixed_NaN_inds = pd.isnull(fixed_locs).nonzero()[0]

#Save as CSV
food_inspections.to_csv('fixed_locations.csv', sep=',')

#Print results
print "The original dataset is missing %s locations." % len(original_NaN_inds) 
print "The fixed dataset is missing %s locations." % len(fixed_NaN_inds)

