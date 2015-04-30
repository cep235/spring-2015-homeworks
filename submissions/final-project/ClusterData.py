import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics

# ============================================================================= #
#                                                                               #
# ABOUT THIS SCRIPT                                                             #
#                                                                               #
# This script attempts to cluster restaurant data using each restaurant's geo-  #
# location pair (longitude, latitude) and percentage of violations failed.      #
#                                                                               #
# Note that this script loads a CSV file called 'fixed_locations.csv'. This CSV #
# file is the same as "Food_Establishment_Inspections.csv", except missing geo- #
# location data has been filed in. (Note, some geolocation data is still miss-  #
# sing, but the number of missing geolocations is very small relative to the    #
# amount of number of missing geolocations in the original dataset.)            #
# ============================================================================= #


''''''''''''''''''''''''''''''''''''
'        F U N C T I O N S         '
''''''''''''''''''''''''''''''''''''

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

def checkDict(rest_info,name,geolocation,count):

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
		if (rest_info[name]['Location'] == geolocation):
			return name #this name is ok

		#Else, we want to add a number at the end of the restaurant name. For example... If
		# "Dunkin Donuts" is already in the dictionary, we want to try changing its name to "Dunkin Donuts 2"
		else:

			if (str(count-1) in name):
				name = name.strip(' ' + str(count-1))

			name = name + ' ' + str(count)

			return checkDict(rest_info,name,geolocation,count+1)
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
		name = "O'Naturals"

	else:
		name = business_name.lower() #Just take the business name and convert it to all lower letters.

	return name


''''''''''''''''''''''''''''''''''''
'     L O A D    D A T A S E T     '
''''''''''''''''''''''''''''''''''''
#Load original and fixed locations CSV
original_dataset = pd.read_csv('Food_Establishment_Inspections.csv', sep=',',low_memory=False)
fixed_dataset = pd.read_csv('fixed_locations.csv', sep=',',low_memory=False)

#Let's compare how many geolocations are missing.
original_locs = original_dataset['Location']
fixed_locs = fixed_dataset['Location']

original_NaN_inds = pd.isnull(original_locs).nonzero()[0]
fixed_NaN_inds = pd.isnull(fixed_locs).nonzero()[0]

print "The original dataset is missing %s locations." % len(original_NaN_inds) 
print "The fixed dataset is missing %s locations." % len(fixed_NaN_inds)


''''''''''''''''''''''''''''''''''''
'  P R E P R O C E S S   D A T A   '
''''''''''''''''''''''''''''''''''''
#Remove NaN rows from the fixed_dataset. (There shouldn't be many)
fixed_dataset = fixed_dataset.drop(fixed_NaN_inds)

#Extract the columns we want
fixed_dataset = fixed_dataset[['BusinessName','RESULT','ViolLevel','Location']]

#Fix violation level: convert {NaN, *, **, ***} to {0, 1, 2, 3}
cleaned_data = fixViolLevel(fixed_dataset)

#Restaurant information will be stored in a dictionary object called 'rest_info'
rest_info = {}

#Now let's go through the entire dataset
for idx, row in cleaned_data.iterrows():

	business_name = row['BusinessName']          #Get name of the business
	result = row['RESULT']                       #Get result of the health inspection
	viol_level = row['ViolLevel']                #Get the violation level
	geolocation = literal_eval(row['Location'])  #Get the restaurant location. Convert from string to tuple

	#Before we add anything to our rest_info dictionary, let's check if the current restaurant is actually in the dictionary.
	#We want to make sure that if we have a restaurant name like "Dunkin Donuts" (which is obviously a chain) that we
	#differentiate it from all the other "Dunkin Donuts" restaurants.
	name = checkDict(rest_info, business_name.lower(), geolocation, 2)


	#Add information for this row into the 'rest_info' dictionary
	if (name not in rest_info):

		#Initialize empty dictionary. (This dictionary will be stored in the 'rest_info' dictionary)
		temp = {}
		temp['ViolLevel'] = []

		#Get pass/fail info
		if ((result == 'HE_Pass') or (result == 'Pass') or (result == 'Passed')):
			temp['Pass'] = 1
			temp['Fail'] = 0
		elif ((result == 'HE_Fail') or (result == 'HE_Filed') or (result == 'Fail') or (result == 'Failed')):
			temp['Pass'] = 0
			temp['Fail'] = 1
		else:
			temp['Pass'] = 0
			temp['Fail'] = 0

		#Now update the total number of health inspections, along with the violation level and geolocation
		temp['Total'] = 1
		temp['ViolLevel'].extend([viol_level])
		temp['Location'] =  geolocation

		rest_info[name] = temp

	else:

		if ((result == 'HE_Pass') or (result == 'Pass') or (result == 'Passed')):
			rest_info[name]['Pass'] += 1
			rest_info[name]['Total'] += 1

		elif ((result == 'HE_Fail') or (result == 'HE_Filed') or (result == 'Fail') or (result == 'Failed')):
			rest_info[name]['Fail'] += 1
			rest_info[name]['Total'] += 1

		else:
			rest_info[name]['Total'] += 1

		rest_info[name]['ViolLevel'].extend([viol_level])

	#Print status!
	print "index: %d,  name: %s" % (idx,name)


''''''''''''''''''''''''''''''''''''
'     C L U S T E R    D A T A     '
''''''''''''''''''''''''''''''''''''
#Convert dictionary to dataframe
rest_info_df = pd.DataFrame.from_dict(rest_info, orient='index')

#Sum up number of violations and get the average
rest_info_df['ViolLevel'] = rest_info_df['ViolLevel'].apply(lambda x: np.mean(x))

#Break geolocation into 2 columns
rest_info_df[['Longitude','Latitude']] = rest_info_df['Location'].apply(pd.Series)
rest_info_df = rest_info_df.drop('Location',axis=1)

#Compute %fail and %pass, then store them in the dataframe
fail_rate = (rest_info_df['Fail'] / rest_info_df['Total']).as_matrix()
pass_rate = (rest_info_df['Pass'] / rest_info_df['Total']).as_matrix()
rest_info_df['Fail Rate'] = fail_rate
rest_info_df['Pass Rate'] = pass_rate

#Choose X 
X = rest_info_df[['ViolLevel','Fail Rate','Pass Rate']]
X = X.as_matrix()
#X = X[:,np.newaxis]

longitude = rest_info_df['Longitude'].as_matrix()
latitude = rest_info_df['Latitude'].as_matrix()

'''
#Decide the number of clusters we should use
error = np.zeros(11)
error[0] = 0;
for k in range(1,11):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit_predict(X)
    error[k] = kmeans.inertia_

plt.plot(range(1,len(error)),error[1:])
plt.xlabel('Number of clusters')
plt.ylabel('Error')


#Looks like we should use 5 clusters!
n_clusters = 5
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
labels = kmeans.fit_predict(X)

#Now let's plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
#colors = ['r','g','b','m','k']
#cmap = sns.color_palette("bright", n_colors=5)

inds1 = np.where(labels == 0)[0].tolist()
inds2 = np.where(labels == 1)[0].tolist()
inds3 = np.where(labels == 2)[0].tolist()
inds4 = np.where(labels == 3)[0].tolist()
inds5 = np.where(labels == 4)[0].tolist()

ax1.scatter(longitude[inds1],latitude[inds1],c=cmap[0],alpha=0.5)
ax1.scatter(longitude[inds2],latitude[inds2],c=cmap[1],alpha=0.5)
ax1.scatter(longitude[inds3],latitude[inds3],c=cmap[2],alpha=0.5)
ax1.scatter(longitude[inds4],latitude[inds4],c=cmap[3],alpha=0.5)
ax1.scatter(longitude[inds5],latitude[inds5],c=cmap[4],alpha=0.5)

plt.show()


cmap = sns.color_palette("bright", n_colors=5)
for jj, label in enumerate(set(labels)):

	#Let's get the indices for the current label
	inds = np.where(labels == label)[0].tolist()

	#...and plot
	ax1.scatter(longitude[inds],latitude[inds],c=cmap[jj],alpha=0.5)

plt.show()
'''

#Looks like we should use 5 clusters!
n_clusters = 5
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
labels = kmeans.fit_predict(X)


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import FoodInspect

fig = plt.figure()
ax = fig.add_subplot(111)
 
mapt = Basemap(projection='merc', lat_0 = 42, lon_0 = -71,
    resolution = 'f', area_thresh = 0.1,
    llcrnrlon=-71.2, llcrnrlat=42.2,
    urcrnrlon=-70.8, urcrnrlat=42.5)
 
mapt.drawcoastlines(ax=ax)
mapt.drawcountries(ax=ax)
mapt.fillcontinents(ax=ax,color='coral',lake_color='aqua')
mapt.drawmapboundary(ax=ax,fill_color='aqua')
mapt.drawstates(ax=ax)
mapt.drawcounties(ax=ax)
mapt.drawmapboundary(ax=ax)
ax.set_axis_bgcolor('aqua')

inds1 = np.where(labels == 0)[0].tolist()
inds2 = np.where(labels == 1)[0].tolist()
inds3 = np.where(labels == 2)[0].tolist()
inds4 = np.where(labels == 3)[0].tolist()
inds5 = np.where(labels == 4)[0].tolist()

x1,y1 = mapt(latitude[inds1],longitude[inds1])
x2,y2 = mapt(latitude[inds2],longitude[inds2])
x3,y3 = mapt(latitude[inds3],longitude[inds3])
x4,y4 = mapt(latitude[inds4],longitude[inds4])
x5,y5 = mapt(latitude[inds5],longitude[inds5])


#x,y = mapt(locations['Latitude'].as_matrix(),locations['Longitude'].as_matrix())
mapt.plot(x1,y1,'rD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x2,y2,'kD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x3,y3,'gD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x4,y4,'mD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x5,y5,'yD',markersize=5,ax=ax,alpha=0.4)
ax.set_title('Restaurants Clustered by Violation Level')

plt.show()

#Plot heatmap
idx = np.argsort(labels)
euclidean_dists = metrics.euclidean_distances(X)
rearranged_dists = euclidean_dists[idx,:][:,idx]
sns.heatmap(rearranged_dists, xticklabels=False, yticklabels=False, linewidths=0, square=True,cbar=False)
plt.show()