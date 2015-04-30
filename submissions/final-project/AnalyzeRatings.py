import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from ast import literal_eval
from matplotlib import cm


# ============================================================================== #
#                                                                                #
# ABOUT THIS SCRIPT                                                              #
#                                                                                #
# This script analyzes all restaurant data from the Food Establishment           #
# Inspections dataset from CityofBoston.gov. The actual data collection piece    #
# takes place in "GetYelpData.py". This script loads the output of GetRatings.py #
# and uses that output to analyze all restaurant data.                           #
# ============================================================================== #



'''''''''''''''''''''''''''''''''
'      L O A D    D A T A       '
'''''''''''''''''''''''''''''''''

#Open our saved pickle
RESTAURANTS = pickle.load( open( "RESTAURANT_INFO.p", "rb" ) )



'''''''''''''''''''''''''''''''''
'   E X T R A C T    D A T A    '
'''''''''''''''''''''''''''''''''

#Convert dictionary to dataframe
rest_info_df = pd.DataFrame.from_dict(RESTAURANTS, orient='index')

#Remove NaN rows
NaN_inds = pd.isnull(rest_info_df['Location']).nonzero()[0]
rest_info_df = rest_info_df.drop(rest_info_df.index[NaN_inds])

#Remove rows whose Yelp ratings are -1. (We want to do a fair comparison. Yelp rating of -1 means we couldn't get a Yelp rating.)
rest_info_df = rest_info_df[rest_info_df['Yelp Rating'] > 0]

#Get Yelp Rating
yelp_rating = rest_info_df['Yelp Rating'].as_matrix()

#Sum up number of violations and get the average
rest_info_df['ViolLevel'] = rest_info_df['ViolLevel'].apply(lambda x: np.mean(x))
avg_violation = rest_info_df['ViolLevel'].as_matrix()

#Break geolocation into 2 columns
rest_info_df['Location'] = rest_info_df['Location'].apply(lambda x: literal_eval(x)) #Convert string to tuple
rest_info_df[['Longitude','Latitude']] = rest_info_df['Location'].apply(pd.Series) #Break tuple into 2 columns
rest_info_df = rest_info_df.drop('Location',axis=1) #Drop the original 'Location' column. We don't need it anymore.

#Compute %fail and %pass, then store them in the dataframe
perc_fail = (rest_info_df['Fail'] / rest_info_df['Total']).as_matrix()
perc_pass = (rest_info_df['Pass'] / rest_info_df['Total']).as_matrix()
rest_info_df['Fail Rate'] = perc_fail
rest_info_df['Pass Rate'] = perc_pass



'''''''''''''''''''''''''''''''''
'    A N A L Y Z E   D A T A    '
'''''''''''''''''''''''''''''''''

# =========== Print Rankings of all the Restaurants =========== #

rest_names = np.asarray(rest_info_df.index) #Each index of 'rest_info_df' is a restaurant name

#Sort restaurants in order of failure!
ids = np.argsort(perc_fail)
ids = ids[::-1]
perc_fail_sorted = perc_fail[ids]
rest_names_sorted = rest_names[ids]


#Print rankings of restaurants
print ""
print "Ranking of Top 100 Restaurants by % Failure:"
for xx in range (0,100):
	print "%d. %s: %3.1f%% (%s)" % (xx+1, rest_names_sorted[xx], 100*perc_fail_sorted[xx],RESTAURANTS[rest_names_sorted[xx]]['LICSTATUS'])


# ============== Get Yelp Data by Neighborhood ================ #
print "\n"
print " ================================="
print "|     Neighborhood Statistics     |"
print " ================================="

avg_viol_lvl_lst = []
avg_yelp_rating_lst = []
avg_failed_ins_lst = []
neighborhood_lst = []


#Group by Neighborhood
neighborhoods_gb = rest_info_df.groupby('Neighborhood')
for neighborhood,df in neighborhoods_gb:

	#Get average violation level, average Yelp rating, and percent health inspections failed
	avg_viol_lvl = df['ViolLevel'].mean()
	avg_yelp_rating = df['Yelp Rating'].mean()
	perc_health_failed = (100*float(df['Fail'].sum()) / df['Total'].sum())

	#Save those values to their respective lists
	neighborhood_lst.extend([neighborhood])
	avg_viol_lvl_lst.extend([avg_viol_lvl])
	avg_yelp_rating_lst.extend([avg_yelp_rating])
	avg_failed_ins_lst.extend([perc_health_failed])

	#Print out results
	print neighborhood
	print "-----------------------"
	print "Avg violation level: %1.2f" % avg_viol_lvl
	print "Avg Yelp rating: %1.2f" % avg_yelp_rating
	print "Avg %% Health Inspect. Failed: %2.1f%%" % perc_health_failed
	print ""



'''''''''''''''''''''''''''''''''
'      P L O T    D A T A       '
'''''''''''''''''''''''''''''''''

ORANGE = '#FF8800'
BLUE = '#0095FF'
GREEN = '#00FF44'

# ==================== Plot Neighborhood Statistics ===================== #
g = plt.figure()
ax = g.add_subplot(111)
x = np.arange(len(neighborhood_lst))
for i in range(0,len(x)):
	ax.barh(x[i],avg_yelp_rating_lst[i],color=cm.jet(1.*i/len(x)),align="center")
ax.set_xlabel('Average Yelp Rating')
ax.set_title('Average Yelp Rating By Neighborhood')
ax.set_ylim([0-0.5,len(x)-0.5])
plt.yticks(x,neighborhood_lst,fontsize=9)
plt.show()


# ========= Plot Yelp Results vs. % Health Inspections Failed =========== #

#Plot Yelp Rating vs. Percent failed
X = yelp_rating.T
X = sm.add_constant(X) #need to add a constant; otherwise, linear regression fails!
y = perc_fail
x_prime = np.asarray([np.linspace(1,5,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,sharex='col',sharey='row')
ax1.scatter(X[:,1],y,alpha=0.05,color=ORANGE)
ax1.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax1.text(0.1, 0.9, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
ax1.set_ylabel('% Health Inspections Failed')
ax1.set_title('Average Yelp User Rating vs.\n % Health Inspections Failed')


#Plot Avg Violation vs. Percent failed
X = avg_violation.T
X = sm.add_constant(X)
y = perc_fail
x_prime = np.asarray([np.linspace(0,3,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax2.scatter(X[:,1],y,alpha=0.15,color=ORANGE)
ax2.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax2.text(2.25, 0.2, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
ax2.set_xlim([0, 3])
ax2.set_ylim([0, 1])
ax2.set_title('Average Restaurant Violation Level vs.\n % Health Inspections Failed')


#Plot Yelp Rating vs. Percent Passed
X = yelp_rating.T
X = sm.add_constant(X)
y = perc_pass
x_prime = np.asarray([np.linspace(1,5,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax3.scatter(X[:,1],y,alpha=0.05,color=BLUE)
ax3.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax3.text(0.13, 0.9, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':GREEN, 'alpha':0.2, 'pad':10})
ax3.set_xlabel('Yelp User Rating')
ax3.set_ylabel('% Health Inspections Passed')
ax3.set_title('Average Yelp User Rating vs.\n % Health Inspections Passed')


#Plot Avg Violation vs. Percent Passed
X = avg_violation.T
X = sm.add_constant(X)
y = perc_pass
x_prime = np.asarray([np.linspace(0,3,10)]).T
x_prime = sm.add_constant(x_prime)

result = sm.OLS(y,X).fit()
y_pred = result.predict(x_prime)
R2 = '{0:.3f}'.format(result.rsquared)

ax4.scatter(X[:,1],y,alpha=0.15,color=BLUE)
ax4.plot(x_prime[:,1],y_pred,'k',linewidth=3)
ax4.text(1.75, 0.8, str('$R^2 = $') + R2, style='italic',bbox={'facecolor':GREEN, 'alpha':0.2, 'pad':10})
ax4.set_xlim([0, 3])
ax4.set_ylim([0, 1])
ax4.set_xlabel('Average Violation Level')
ax4.set_title('Average Restaurant Violation Level vs.\n % Health Inspections Passed')
plt.show()