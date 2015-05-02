import Yelp
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import chain
import sys

# ============================================================================= #
#                                                                               #
# ABOUT THIS SCRIPT                                                             #
#                                                                               #
# This script makes calls to the official Yelp API to obtain the following inf- #
# ormation about each restaurant:                                               #
#      (1) Average Yelp Rating                                                  #
#      (2) Number of Yelp reviews                                               #
#      (3) The neighborhood the restaurant is located in (e.g., Roxbury)        #
#      (4) Category of food (e.g., Spanish, Italian, Mexican, Japanese, etc.)   #
#                                                                               #
# ============================================================================= #

#Keep track of time
start_time = time.time()


''''''''''''''''''''''''''''''''''''
'        F U N C T I O N S         '
''''''''''''''''''''''''''''''''''''

def checkDict(rest_info,name,address,count):

	'''
	Some restaurants have the same name. For example, there are many restaurants with the
	name "Dunkin Donuts". Ideally, we want to distinguish between each Dunkin Donuts restaurant.
	Therefore, this function attempts to rename restaurants with an appropriate name. For example,
	if there are 4 Dunkin Donuts restaurants in the dataset, we want to name them as Dunkin Donuts,
	Dunkin Donuts 2, Dunkin Donuts 3, and Dunkin Donuts 4.


	Input:
		@param rest_info: Dictionary of restaurant information
		@param name: Restaurant name to be checked
		@param geolocation: Restaurant geolocation to be checked
		@param count: Used for generating a new name for the restaurant

	Returns:
		@return name of the restaurant. (Note: if input name is already taken,
			then 'name' is renamed)

	'''

	#First check if the name is in the 'RESTAURANTS' dictionary
	if (name in rest_info):

		#If it is, then let's verify the address
		if (rest_info[name]['Address'] == address):
			return name #this name is ok

		#Else, we want to add a number at the end of the restaurant name. For example... If
		# "Dunkin Donuts" is already in the dictionary, we want to try changing its name to "Dunkin Donuts 2"
		else:

			if (str(count-1) in name):
				name = name.strip(' ' + str(count-1))

			name = name + ' ' + str(count)

			return checkDict(rest_info,name,address,count+1)
	else:
		return name



def adjustName(business_name):

	'''
	Some restaurants have names which confuse Yelp. I've found these particular names through
	trial and error. Therefore, this simple function just adjusts the name of the restaurant,
	as needed.

	Inputs:
		@param business_name: Name of the business, as extracted from the dataset

	Returns:
		@return: Adjusted business name
	'''

	#Just a few exceptions... found by trial and error.
	#For some reason, these business names confuse the hell out of Yelp
	if (business_name == '20TH CENTRY BOWLING LANES'):
		name = 'rons gourmet ice cream and twentieth century bowl';

	elif (business_name == '412 Broadway'): #I had to look up this real restaurant's name. Apparently it's a taco and oyster place.
		name = 'loco taqueria & oyster bar'

	elif (business_name == '21 ST. AMENDMENT'):
		name = '21st amendment'

	elif (business_name == '224 BOSTON STREET'):
		name = '224 boston street restaurant'

	elif (business_name == '68 Chinese Fast Food'):
		name = '68 chinese'

	elif (business_name == '75 CHESTNUT'):
		name = '75 chestnut'

	elif (business_name == "FOUR'S BOSTON"):
		name = "the four's"

	elif (business_name.startswith('DUNKIN DONUTS')): 
		name = 'dunkin donuts'

	elif (business_name == "brigham & womens hospital d/b/a o'naturals"):
		name = "o'naturals"

	elif (business_name == "shriners' hosp. for children"):
		name = "shriners hospital for children"

	else:
		name = business_name.lower() #Just take the business name and convert it to all lower letters.

	return name


def fixViolLevel(food_inspections):

	'''
	This function converts violation level (NaN, *, **, and ***) to numerical values. Specifically,
		NaN  ->  0
		*    ->  1
		**   ->  2
		***  ->  3
	'''
	LE = LabelEncoder()
	food_inspections['ViolLevel'] = food_inspections['ViolLevel'].fillna(0)
	food_inspections['ViolLevel'] = LE.fit_transform(food_inspections['ViolLevel'])

	return food_inspections


''''''''''''''''''''''''''''''''''''
' O B T A I N   Y E L P   D A T A  '
''''''''''''''''''''''''''''''''''''

#First check if we can scrape data from Yelp!
try:
	results = Yelp.search('food','Boston MA')
except:
	print "ERROR: Could not extract data from Yelp. Looks like Yelp was unable to authenticate you. Do you have the necessary authentication tokens and keys? Check Yelp.py for more information."
	sys.exit()

#Load data
food_inspections = pd.read_csv('fixed_locations.csv', sep=',',low_memory=False)
food_inspections = food_inspections.iloc[::-1]

#Convert the set {NaN,*,**,***} in ViolLevel to the set {0,1,2,3}
food_inspections = fixViolLevel(food_inspections)

#Drop the unnecessary columns
columns = ['DBAName','LegalOwner','NameLast','NameFirst','LICENSENO','ISSDTTM','EXPDTTM','DESCRIPT','RESULTDTTM','Violation','ViolDesc','VIOLDTTM','StatusDate','Comments','State','Zip','Property_ID']
food_inspections = food_inspections.drop(columns,axis=1)


#Get all the unique business names and the total number of passed / failed health inspections. 
RESTAURANTS = {} #Store all restaurant info in this dict

for idx, row in food_inspections.iterrows():

	#Get name of the business
	business_name = row['BusinessName']

	#Check status of license
	license_status = row['LICSTATUS']

	#Check if they passed or failed the health inspection
	result = row['RESULT']

	#Get the violation level
	viol_level = row['ViolLevel']

	#Adjust the name of the business
	name = adjustName(business_name)

	#Get business address
	try:
		address =  row['Address'] + ' ' + row['City'] + ' MA'
	except:
		continue #This is the case where the city or address is nan

	address = address.lower()

	#Check if the name is actually in the dictionary
	name = checkDict(RESTAURANTS,name,address,2)

	#Add info to food inspections
	if (name not in RESTAURANTS):

		temp = {}
		temp['ViolLevel'] = []

		#Get pass/fail info
		if ((result == 'HE_Pass') or (result == 'Pass') or (result == 'Passed') or (result == 'HE_Filed')):
			temp['Pass'] = 1
			temp['Fail'] = 0
		elif ((result == 'HE_Fail') or (result == 'Fail') or (result == 'Failed')):
			temp['Pass'] = 0
			temp['Fail'] = 1
		else:
			temp['Pass'] = 0
			temp['Fail'] = 0

		#Now update the total number of health inspections, along with the license status & address
		temp['Total'] = 1
		temp['LICSTATUS'] = row['LICSTATUS']
		temp['License Type'] = row['LICENSECAT']
		temp['ViolLevel'].extend([viol_level])
		temp['Address'] = address
		temp['Location'] = row['Location']


		#Get Yelp rating for the current business.
		try:
			results = Yelp.search(name,address)
		except:
			temp['Yelp Rating'] = -1
			RESTAURANTS[name] = temp #Couldn't get Yelp info...
			continue

		time.sleep(2) #We need to wait between API requests. (Might as well put the 'sleep' command here.)

		try:
			business_info = results['businesses']
		except:
			business_info = results['businesses'][0]

		#Check if Yelp's response is empty. If so, jump to the next loop iteration
		if (len(business_info) == 0):
			temp['Yelp Rating'] = -1
			temp['Neighborhood'] = None
			temp['Review Count'] = -1
			temp['Categories'] = None
			RESTAURANTS[name] = temp #Couldn't get Yelp info...
			continue
		
		#Store Yelp rating &  review count
		try:
			temp['Yelp Rating'] = float(business_info[0]['rating'])
			temp['Review Count'] = business_info[0]['review_count']

			#Extract neighborhood. (Sometimes neighborhood is not given.)
			try:
				temp['Neighborhood'] = business_info[0]['location']['neighborhoods'][0]
			except KeyError:
				temp['Neighborhood'] = None

			#Extract category of food. (Sometimes this is not given.)
			try:
				cats = business_info[0]['categories'] #Obtain category data, which is saved as a list of lists
				cats = list(chain(*cats)) #Convert list of lists into 1 list
				cats = cats[0::2] #Grab only the odd indices in the list. Every even index is a duplicate.
				temp['Categories'] = cats
			except KeyError:
				temp['Categories'] = None

		#Sometimes Yelp is inconsistent with the way their API returns data... So this 'except' case handles those rare situations
		except TypeError, IndexError:
			try:
				temp['Yelp Rating'] = float(business_info['rating'])
				temp['Review Count'] = business_info['review_count']

				#Extract neighborhood. (Sometimes neighborhood is not given.)
				try:
					temp['Neighborhood'] = business_info['location']['neighborhoods'][0]
				except KeyError:
					temp['Neighborhood'] = None

				#Extract category of food. (Sometimes this is not given.)
				try:
					cats = business_info['categories'] #Obtain category data, which is saved as a list of lists
					cats = list(chain(*cats)) #Convert list of lists into 1 list
					cats = cats[0::2] #Grab only the odd indices in the list. Every even index is a duplicate.
					temp['Categories'] = cats
				except KeyError:
					temp['Categories'] = None

			#This is the case where we can't get any information from Yelp. All hope is lost
			except TypeError, IndexError:
				temp['Yelp Rating'] = -1 #We really can't get a Yelp rating :(
				temp['Neighborhood'] = None
				temp['Review Count'] = -1
				temp['Categories'] = None

		#Update restaurants dict
		RESTAURANTS[name] = temp

	else:

		if ((result == 'HE_Pass') or (result == 'Pass') or (result == 'Passed')):
			RESTAURANTS[name]['Pass'] += 1
			RESTAURANTS[name]['Total'] += 1

		elif ((result == 'HE_Fail') or (result == 'HE_Filed') or (result == 'Fail') or (result == 'Failed')):
			RESTAURANTS[name]['Fail'] += 1
			RESTAURANTS[name]['Total'] += 1

		else:
			RESTAURANTS[name]['Total'] += 1

		RESTAURANTS[name]['ViolLevel'].extend([viol_level])


	#Print status!
	print "index: %d,  name: %s,  rating: %1.1f" % (idx,name,RESTAURANTS[name]['Yelp Rating'])



#Now save the results as a pickle file
pickle.dump(RESTAURANTS, open( "RESTAURANT_INFO.p", "wb" ) )

print("This script took %s seconds to run." % (time.time() - start_time))