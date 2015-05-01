import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import pickle

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

#Open our saved pickle
rest_info = pickle.load( open( "RESTAURANT_INFO.p", "rb" ) )



''''''''''''''''''''''''''''''''''''
'     P R O C E S S    D A T A     '
''''''''''''''''''''''''''''''''''''
#Convert dictionary to dataframe
rest_info_df = pd.DataFrame.from_dict(rest_info, orient='index')

#Remove NaN rows
NaN_inds = pd.isnull(rest_info_df['Location']).nonzero()[0]
rest_info_df = rest_info_df.drop(rest_info_df.index[NaN_inds])

#Sum up number of violations and get the average
rest_info_df['ViolLevel'] = rest_info_df['ViolLevel'].apply(lambda x: np.mean(x))

#Break geolocation into 2 columns
rest_info_df['Location'] = rest_info_df['Location'].apply(lambda x: literal_eval(x)) #Convert string to tuple
rest_info_df[['Longitude','Latitude']] = rest_info_df['Location'].apply(pd.Series) #Break tuple into 2 columns
rest_info_df = rest_info_df.drop('Location',axis=1) #Drop the original 'Location' column. We don't need it anymore.

#Compute %fail and %pass, then store them in the dataframe
perc_fail = (rest_info_df['Fail'] / rest_info_df['Total']).as_matrix()
perc_pass = (rest_info_df['Pass'] / rest_info_df['Total']).as_matrix()
rest_info_df['Fail Rate'] = perc_fail
rest_info_df['Pass Rate'] = perc_pass

#Choose X 
X = rest_info_df[['ViolLevel','Fail Rate','Pass Rate']]
X = X.as_matrix()
#X = X[:,np.newaxis]

longitude = rest_info_df['Longitude'].as_matrix()
latitude = rest_info_df['Latitude'].as_matrix()


''''''''''''''''''''''''''''''''''''
'     C L U S T E R    D A T A     '
''''''''''''''''''''''''''''''''''''

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
plt.title('Number of Clusters vs. Error')


#Looks like we should use 3 clusters!
n_clusters = 3
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
labels = kmeans.fit_predict(X)


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
 
mapt = Basemap(projection='merc', lat_0 = 42, lon_0 = -71,
    resolution = 'f', area_thresh = 0.1,
    llcrnrlon=-71.2, llcrnrlat=42.2,
    urcrnrlon=-70.8, urcrnrlat=42.5)
 
mapt.drawcoastlines(ax=ax)
mapt.drawcountries(ax=ax)
mapt.fillcontinents(ax=ax,color='lightgreen',lake_color='aqua')
mapt.drawmapboundary(ax=ax,fill_color='aqua')
mapt.drawstates(ax=ax)
mapt.drawcounties(ax=ax)
mapt.drawmapboundary(ax=ax)
ax.set_axis_bgcolor('aqua')

inds1 = np.where(labels == 0)[0].tolist()
inds2 = np.where(labels == 1)[0].tolist()
inds3 = np.where(labels == 2)[0].tolist()
#inds4 = np.where(labels == 3)[0].tolist()
#inds5 = np.where(labels == 4)[0].tolist()

x1,y1 = mapt(latitude[inds1],longitude[inds1])
x2,y2 = mapt(latitude[inds2],longitude[inds2])
x3,y3 = mapt(latitude[inds3],longitude[inds3])
#x4,y4 = mapt(latitude[inds4],longitude[inds4])
#x5,y5 = mapt(latitude[inds5],longitude[inds5])


#x,y = mapt(locations['Latitude'].as_matrix(),locations['Longitude'].as_matrix())
mapt.plot(x1,y1,'rD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x2,y2,'kD',markersize=5,ax=ax,alpha=0.4)
mapt.plot(x3,y3,'cD',markersize=5,ax=ax,alpha=0.4)
#mapt.plot(x4,y4,'mD',markersize=5,ax=ax,alpha=0.4)
#mapt.plot(x5,y5,'yD',markersize=5,ax=ax,alpha=0.4)
ax.set_title('Restaurant Clustering #1')
ax.legend(['Cluster 1','Cluster 2','Cluster 3']) #,'Cluster 4','Cluster 5'])

plt.show()

#Plot heatmap
idx = np.argsort(labels)
euclidean_dists = metrics.euclidean_distances(X)
print X.shape
print euclidean_dists.shape
rearranged_dists = euclidean_dists[idx,:][:,idx]
print rearranged_dists.shape
sns.heatmap(rearranged_dists, xticklabels=False, yticklabels=False, linewidths=0, square=True,cbar=False)
plt.show()