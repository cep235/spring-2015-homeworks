import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


# ============================================================================= #
#                                                                               #
# ABOUT THIS SCRIPT                                                             #
#                                                                               #
# This script analyzes hopsital restaurant data from the Food Establishment     #
# Inspections dataset from CityofBoston.gov. The actual data collection piece   #
# takes place in "GetRatings.py". This script loads the output of GetRatings.py #
# and uses that output to analyze hospital data                                 #
# ============================================================================= #


''''''''''''''''''''''''''''''''''''''''''''''''
'              L O A D    D A T A              '
''''''''''''''''''''''''''''''''''''''''''''''''

#Open our saved pickle
RESTAURANTS = pickle.load( open( "RESTAURANT_INFO.p", "rb" ) )
print "Data loaded."


''''''''''''''''''''''''''''''''''''''''''''''''
' C H E C K   H O S P I T A L   R A T I N G S  '
''''''''''''''''''''''''''''''''''''''''''''''''
hospital_count = 0
for restaurant in RESTAURANTS:
	if ('hosp' in restaurant):
		hospital_count += 1

		#Prints out the hospital restaurant name along with its Yelp rating. A Yelp rating of -1 
		#means we couldn't find a Yelp rating for that restaurant.
		print "hospital: %s,   rating: %1.1f" % (restaurant, RESTAURANTS[restaurant]['Yelp Rating'])


print "\nFound %d hospitals in the food inspections dataset." % hospital_count

# ============================================================================= #
#                                                                               #
# NOTE                                                                          #
#                                                                               #
# We see that some hospitals do not have a Yelp rating. Since there aren't many #
# hospitals that are missing a Yelp rating, I've decided to manually look them  #
# up. (We're only missing 8 hospital restaurant ratings.)                       #
#                                                                               #
# It turns out that Yelp couldn't find some of these restaurant ratings for one #
# of two reasons: (1) The restaurant's name changed, or (2) The restaurant was  #
# never added to Yelp's database. (Yelp still maintains information for         #
# restaurants that went out of business. That means if a restaurant does not    #
# exist in their database, then it was never added.)                            #
#                                                                               #
# To determine if a restaurant's name was changed or if the restaurant was      #
# never added, I went to each hospital's website and checked their restaurant   #
# information. I also went to Yelp and searched for "food" near the addresses   #
# of the restaurants which were missing Yelp ratings. For example, "The Plaza   #
# Cafe" no longer exists at Massachusetts General Hospital, but its new name is #
# "Eat Street Cafe". We can confirm this namge change by looking at Mass. Gen.  #
# Hospital's website and looking up this information on Yelp.                   #
# ============================================================================= #


''''''''''''''''''''''''''''''''''''''''''''''''
'  E X T R A C T    H O S P I T A L    D A T A '
''''''''''''''''''''''''''''''''''''''''''''''''

#List of hospitals. Obtained from: http://en.wikipedia.org/wiki/List_of_hospitals_in_Massachusetts
hospitals=[
"Arbour",
"Boston Medical Center",
"Carney",
"Faulkner",
"Franciscan",
"Kindred",
"Hebrew Rehabilitation",
"Jewish Memorial",
"Lemuel Shattuck",
"Lindemann",
"Massachusetts Eye and Ear Infirmary",
"Mass Eye and Ear Infirmary",
"Massachusetts General Hospital",
"Mattapan Community Health Center",
"New England Baptist Hospital",
"Solomon Mental",
"St. Elizabeth",
"St. Margaret",
"Jude",
"Shriner",
"Spaulding",
"Tuft",
"VA Hospital",
"Beth Israel",
"Deaconess",
"Boston Children",
"Brigham and Women",
"Dana-Farber",
"Joslin Diabetes",
"hosp"]

#Convert hospitals to all lowercase
for jj in range (0,len(hospitals)):
	hospitals[jj] = hospitals[jj].lower()


avg_violation = [] #Stores the average violation level for each restaurant
perc_failed = [] #Stores the percentage of failed health inspections
perc_passed = [] #Stores the percentage of passed health inspections
yelp_rating = []
hosp_names = [] #List of hospital names

for restaurant in RESTAURANTS:

	#Check if it is a hospital
	is_hospital = False
	for hosp in hospitals:
		if (hosp in restaurant):
			is_hospital = True
			break;



	if (is_hospital):
		total = float(RESTAURANTS[restaurant]['Total'])
		num_passed = float(RESTAURANTS[restaurant]['Pass']) #number of passed inspections
		num_failed = float(RESTAURANTS[restaurant]['Fail']) #number of failed inspections
		viol_level_avg = np.mean(RESTAURANTS[restaurant]['ViolLevel'])

		#============== Just some weird exceptions that I've encountered ============#

		#http://www.brighamandwomens.org/patients_visitors/patientresources/default.aspx
		if (restaurant == "brigham & womens hospital d/b/a o'naturals"):
			try:
				results = Yelp.search("O'Naturals",RESTAURANTS[restaurant]['Address'])
				rating = float(results['businesses'][0]['rating'])
			except:
				rating = -1

		#This restaurant underwent a name change. It's now called "Eat Street Cafe"
		elif (restaurant == "mass gen.hosp./the plaza cafe"): 
			try:
				results = Yelp.search("Eat Street Cafe",RESTAURANTS[restaurant]['Address'])
				rating = float(results['businesses'][0]['rating'])
			except:
				rating = -1

		#This restaurant underwent a name change
		elif (restaurant == 'mass gen. hosp./tea leaves & coffee beans'):
			try:
				results = Yelp.search("Coffee Central",RESTAURANTS[restaurant]['Address'])
				rating = float(results['businesses'][0]['rating'])
			except:
				rating = -1

		elif (restaurant == "the atrium cafe@brigham & womens faulkner hosp."):
			continue #This restaurant doesn't seem to exist anymore and is missing a lot of info

		elif (restaurant == 'brigham & womens hospital l1 o.r. lounge'):
			continue #This restaurant doesn't seem to exist anymore and is missing a lot of info

		elif (restaurant == 'sea to you sushi @ beth israel hospital east campus'):
			continue #This restaurant doesn't seem to exist anymore and is missing a lot of info

		elif (restaurant == 'gourmet bean/faulkner hosp.'):
			continue #This restaurant doesn't seem to exist anymore and is missing a lot of info

		elif (restaurant == "shriners' hosp. for children"):
			try:
				results = Yelp.search("Shriners Hospital for Children",RESTAURANTS[restaurant]['Address'])
				rating = float(results['businesses'][0]['rating'])
			except:
				rating = -1
		else:
			rating = float(RESTAURANTS[restaurant]['Yelp Rating']) #get yelp rating

		PASS = num_passed / total
		FAIL = num_failed / total

		#Only save data points for hospitals which have a Yelp user rating
		if (rating > 0):
			hosp_names.extend([restaurant])
			avg_violation.extend([viol_level_avg])
			perc_failed.extend([FAIL])
			perc_passed.extend([PASS])
			yelp_rating.extend([rating])


#Now get the overall averages
Overall_perc_failed = np.mean(perc_failed)
Overall_perc_passed = np.mean(perc_passed)
Overall_avg_violation = np.mean(avg_violation)

#Convert everything to numpy arrays
perc_failed = np.asarray([perc_failed])
perc_passed = np.asarray([perc_passed])
yelp_rating = np.asarray([yelp_rating])
avg_violation = np.asarray([avg_violation])
hosp_names = np.asarray(hosp_names)


#Sort hospitals in order of failure!
ids = np.argsort(perc_failed[0])
ids = ids[::-1]
perc_failed_sorted = perc_failed[0][ids]
hosp_names_sorted = hosp_names[ids]


print ""
print "Ranking of Hospitals by % Failure:"
for xx in range (0,len(hosp_names)-1):
	print "%s: %3.1f%%" % (hosp_names_sorted[xx], 100*perc_failed_sorted[xx])



'''''''''''''''''''''''''''''''''
'      P L O T    D A T A       '
'''''''''''''''''''''''''''''''''

ORANGE = '#FF8800'
BLUE = '#0095FF'
GREEN = '#00FF44'

#Plot Yelp Rating vs. Percent failed
X = yelp_rating.T
X = sm.add_constant(X) #need to add a constant; otherwise, linear regression fails!
y = perc_failed[0]
x_prime = np.asarray([np.linspace(1,5,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
ax1.scatter(X[:,1],y,alpha=0.25,color=ORANGE)
ax1.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax1.text(0.1, 0.9, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
ax1.set_ylabel('% Health Inspections Failed')
ax1.set_title('Average Yelp User Rating vs.\n % Health Inspections Failed')


#Plot Avg Violation vs. Percent failed
X = avg_violation.T
X = sm.add_constant(X)
y = perc_failed[0]
x_prime = np.asarray([np.linspace(0,3,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax2.scatter(X[:,1],y,alpha=0.25,color=ORANGE)
ax2.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax2.text(2.25, 0.2, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
ax2.set_xlim([0, 3])
ax2.set_ylim([0, 1])
ax2.set_title('Average Restaurant Violation Level vs.\n % Health Inspections Failed')


#Plot Yelp Rating vs. Percent Passed
X = yelp_rating.T
X = sm.add_constant(X)
y = perc_passed[0]
x_prime = np.asarray([np.linspace(1,5,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax3.scatter(X[:,1],y,alpha=0.25,color=BLUE)
ax3.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax3.text(0.13, 0.9, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':GREEN, 'alpha':0.2, 'pad':10})
ax3.set_xlabel('Yelp User Rating')
ax3.set_ylabel('% Health Inspections Passed')
ax3.set_title('Average Yelp User Rating vs.\n % Health Inspections Passed')


#Plot Avg Violation vs. Percent Passed
X = avg_violation.T
X = sm.add_constant(X)
y = perc_passed[0]
x_prime = np.asarray([np.linspace(0,3,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax4.scatter(X[:,1],y,alpha=0.25,color=BLUE)
ax4.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax4.text(1.75, 0.8, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':GREEN, 'alpha':0.2, 'pad':10})
ax4.set_xlim([0, 3])
ax4.set_ylim([0, 1])
ax4.set_xlabel('Average Violation Level')
ax4.set_title('Average Restaurant Violation Level vs.\n % Health Inspections Passed')
plt.show()
