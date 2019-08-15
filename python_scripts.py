#1
"Outer:5 Inner:1"
"Outer:5 Inner:2"
"Outer:4 Inner:1"
"Outer:4 Inner:2"
"Outer:3 Inner:1"
"Outer:3 Inner:2"
"Outer:2 Inner:1"
"Outer:2 Inner:2"


#2
"Outer:1 Inner:1"
"Outer:1 Inner:2"
"Outer:2 Inner:1"
"Outer:2 Inner:2"
"Outer:3 Inner:1"
"Outer:3 Inner:2"
"Outer:4 Inner:1"
"Outer:4 Inner:2"


#3



##########BATTLESHIP CODE##########


from random import randint

board = []

for x in range(0, 5):
  board.append(["O"] * 5)

def print_board(board):
  for row in board:
    print " ".join(row)

print_board(board)

def random_row(board):
  return randint(0, len(board) - 1)

def random_col(board):
  return randint(0, len(board[0]) - 1)

ship_row = random_row(board)
ship_col = random_col(board)
print ship_row
print ship_col
guess_row = int(raw_input("Guess Row: "))
guess_col = int(raw_input("Guess Col: "))

# Write your code below!
if guess_row == ship_row and guess_col == ship_col:
  print "Congratulations! You sank my battleship!"  
elif guess_row == "X" and guess_col == "X":      
  print "You guessed that one already."
else:
  if guess_row not in range(0,ship_row-1) or guess_col not in range(0,ship_col-1):
    print( "Oops, that's not even in the ocean.")
  else:
    	print "You missed my battleship!"

  
  board[guess_row][guess_col] = "X"
  print_board(board)





##########BATTLESHIP CODE w/ 4 turns##########





from random import randint

board = []

for x in range(5):
  board.append(["O"] * 5)

def print_board(board):
  for row in board:
    print " ".join(row)

print_board(board)

def random_row(board):
  return randint(0, len(board) - 1)

def random_col(board):
  return randint(0, len(board[0]) - 1)

ship_row = random_row(board)
ship_col = random_col(board)
print ship_row
print ship_col

for turn in range(4):
    guess_row = int(raw_input("Guess Row: "))
    guess_col = int(raw_input("Guess Col: "))
    print (turn + 1)

if guess_row == ship_row and guess_col == ship_col:
  print "Congratulations! You sunk my battleship!"
else:
  if (guess_row < 0 or guess_row > 4) or (guess_col < 0 or guess_col > 4):
    print "Oops, that's not even in the ocean."
  elif(board[guess_row][guess_col] == "X"):
    print "You guessed that one already."
  else:
    print "You missed my battleship!"
    board[guess_row][guess_col] = "X"

   
  print_board(board) 





##########BATTLESHIP CODE w/ 4 turns and "Game Over"##########






from random import randint

board = []

for x in range(0, 5):
  board.append(["O"] * 5)

def print_board(board):
  for row in board:
    print " ".join(row)

print_board(board)

def random_row(board):
  return randint(0, len(board) - 1)

def random_col(board):
  return randint(0, len(board[0]) - 1)

ship_row = random_row(board)
ship_col = random_col(board)
print ship_row
print ship_col

# Everything from here on should be in your for loop
# don't forget to properly indent!
for turn in range(4):
  print "Turn", turn + 1
  guess_row = int(raw_input("Guess Row: "))
  guess_col = int(raw_input("Guess Col: "))

  if guess_row == ship_row and guess_col == ship_col:
    print "Congratulations! You sank my battleship!"   
  else:
    if guess_row not in range(5) or \
      guess_col not in range(5):
      print "Oops, that's not even in the ocean."
    elif board[guess_row][guess_col] == "X":
      print( "You guessed that one already." )
    else:
      print "You missed my battleship!"
      board[guess_row][guess_col] = "X"
    if (turn == 3):
      print "Game Over"
    print_board(board)


#############################

python for loop (indexes and values)

areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for index, area in enumerate(areas) :
    print("room " + str(index) + ": " + str(area))


############################

LOOP OVER DATAFRAME

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for lab,row in cars.iterrows() :
    print(lab)
    print(row)


############################


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
# for lab, row in cars.iterrows() :
#     cars.loc[lab, "COUNTRY"] = row["country"].upper()
cars["COUNTRY"] = cars["country"].apply(str.upper)

print(cars)

############################

DATAfFRAME adding a column


# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows():
    cars.loc[lab, "COUNTRY"] = (row["country"].upper())


# Print cars
print(cars)

############################

DATAFRAME adding a column using apply(less cost)

# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Use .apply(str.upper)
# for lab, row in cars.iterrows() :
#     cars.loc[lab, "COUNTRY"] = row["country"].upper()
cars["COUNTRY"] = cars["country"].apply(str.upper)

print(cars)

############################

Implement Clumsiness

# numpy and matplotlib imported, seed set

# Simulate random walk 250 times
all_walks = []
for i in range(250) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)

        # Implement clumsiness
        if np.random.rand() <= 0.001 :
            step = 0

        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))
plt.plot(np_aw_t)
plt.show()

############################

PLOT THE DISTRIBUTION

# numpy and matplotlib imported, seed set

# Simulate random walk 500 times
all_walks = []
for i in range(500) :
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)
        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        if np.random.rand() <= 0.001 :
            step = 0
        random_walk.append(step)
    all_walks.append(random_walk)

# Create and plot np_aw_t
np_aw_t = np.transpose(np.array(all_walks))

# Select last row from np_aw_t: ends
ends = np_aw_t[-1,:]

# Plot histogram of ends, display plot
plt.hist(ends)
plt.show()


############################

PERCENTAGE of RUNS OVER A NUMBER IN A SET OF RANDOM RUNS (in shell)

print(np.means((##data field name##) >= (##arbitrary number used##))  = float of 1

############################

Functions that return multiple values

def shout_all(word1, word2):
    """Return a tuple of strings"""
    # Concatenate word1 with '!!!': shout1
    shout1 = word1 + '!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2 = word2 + '!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words = (shout1, shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
yell1, yell2 = shout_all('congratulations', 'you')

# Print yell1 and yell2
print(yell1)
print(yell2)

############################

Map() and LAMBDA FUNCTIONS


# Create a list of strings: spells

spells = ['protego', 'accio', 'expecto patronum', 'legilimens']

# Use map() to apply a lambda function over spells: shout_spells
shout_spells = map(lambda item: item + '!!!', spells)

# Convert shout_spells to a list: shout_spells_list
shout_spells_list = list(shout_spells)

# Print the result
print(shout_spells_list)


############################

Filter and LAMBDA 

# Create a list of strings: 
fellowship
fellowship = ['frodo', 'samwise', 'merry', 'pippin', 'aragorn', 'boromir', 'legolas', 'gimli', 'gandalf']


# Use filter() to apply a lambda function over fellowship: result

result = filter(lambda member: len(member) > 6, fellowship)

# Convert result to a list: result_list
result_list = list(result)

# Print result_list
print(result_list)


############################

Handling exceptions and the filter function with Lambda

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2] == 'RT',tweets_df['text'])

# Create list from filter object result: res_list
res_list = list(result)

# Print all retweets in res_list
for tweet in res_list:
    print(tweet)


############################


Iterating over iterables 

# Create an iterator for range(3): small_value
small_value = iter(range(3))

# Print the values in small_value
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Loop over range(3) and print the values
for num in range(3):
    print(num)

# Create an iterator for range(10 ** 100): googol
googol = iter(range(10 ** 100))

# Print the first 5 values from googol
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))
print(next(googol))


############################


Iterators as function arguments

# Create a range object: values
values = range(10,21)

# Print the range object
print(values)

# Create a list of integers: values_list
values_list = list(values)

# Print values_list
print(values_list)

# Get the sum of values: values_sum
values_sum = sum(values)

# Print values_sum
print(values_sum)


############################

zip and unpack tuples with * 

# Create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# # Print the tuples in z1 by unpacking with *
print(*z1)

# Re-create a zip object from mutants and powers: z1
z1 = zip(mutants, powers)

# 'Unzip' the tuples in z1 by unpacking with * and zip(): result1, result2
result1, result2 = zip(*z1)

# Check if unpacked tuples are equivalent to original tuples
print(result1 == mutants)
print(result2 == powers)


############################

Extracting information for large amounts of Twitter data

# Define count_entries()
def count_entries(csv_file, c_size, colname):
    """Return a dictionary with counts of
    occurrences as value for each key."""
    
    # Initialize an empty dictionary: counts_dict
    counts_dict = {}

    # Iterate over the file chunk by chunk
    for chunk in pd.read_csv(csv_file, chunksize=c_size):

        # Iterate over the column in DataFrame
        for entry in chunk[colname]:
            if entry in counts_dict.keys():
                counts_dict[entry] += 1
            else:
                counts_dict[entry] = 1

    # Return counts_dict
    
return counts_dict

# Call count_entries(): result_counts
result_counts = count_entries('tweets.csv', 10, 'lang')

# Print result_counts
print(result_counts)


############################

Nested List Comprehensions (matrices)

# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]

# Print the matrix
for row in matrix:
    print(row)


############################


Writing a generator to load data in chunks

# Define read_large_file()
def read_large_file(file_object):
    """A generator function to read a large file lazily."""

    # Loop indefinitely until the end of the file
    while True:
        # yield data

        # Read a line from the file: data
        data = file_object.readline()

        # Break if this is the end of the file
        if not data:
            break

        # Yield the line of data
        yield data
        
# Open a connection to the file
with open('world_dev_ind.csv') as file:

    # Create a generator object for the file: gen_file
    gen_file = read_large_file(file)

    # Print the first three lines of the file
    print(next(gen_file))
    print(next(gen_file))
    print(next(gen_file))

############################

List of tuples item chunk

# Code from previous exercise
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)
df_urb_pop = next(urb_pop_reader)
df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']
pops = zip(df_pop_ceb['Total Population'], 
           df_pop_ceb['Urban population (% of total)'])
pops_list = list(pops)

# Use list comprehension to create new DataFrame column 'Total Urban Population'
df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]

# Plot urban population data
df_pop_ceb.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


############################

Writing an iterator to load data chunks

# Initialize reader object: urb_pop_reader
urb_pop_reader = pd.read_csv('ind_pop_data.csv', chunksize=1000)

# Initialize empty DataFrame: data
data = pd.DataFrame()

# Iterate over each DataFrame chunk
for df_urb_pop in urb_pop_reader:

    # Check out specific country: df_pop_ceb
    df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == 'CEB']

    # Zip DataFrame columns of interest: pops
    pops = zip(df_pop_ceb['Total Population'],
                df_pop_ceb['Urban population (% of total)'])

    # Turn zip object into list: pops_list
    pops_list = list(pops)

    # Use list comprehension to create new DataFrame column 'Total Urban Population'
    df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
    # Append DataFrame chunk to data: data
    data = data.append(df_pop_ceb)

# Plot urban population data
data.plot(kind='scatter', x='Year', y='Total Urban Population')
plt.show()


############################


!!!!!!Data SET c/o 

!!!!!!https://www.kaggle.com/worldbank/world-development-indicators



Writing an iterator to load data chunks continued


# Define plot_pop()


def plot_pop(filename, country_code):

    

# Initialize reader object: urb_pop_reader
    

urb_pop_reader = pd.read_csv(filename, chunksize=1000)

    

# Initialize empty DataFrame: data
    

data = pd.DataFrame()
    
    

# Iterate over each DataFrame chunk
    

for df_urb_pop in urb_pop_reader:
        

# Check out specific country: df_pop_ceb
        

df_pop_ceb = df_urb_pop[df_urb_pop['CountryCode'] == country_code]

        

# Zip DataFrame columns of interest: pops
        
pops = zip(df_pop_ceb['Total Population'],
 df_pop_ceb['Urban population (% of total)'])

 
# Turn zip object into list: pops_list

        pops_list = list(pops)

        
# Use list comprehension to create new DataFrame column 'Total Urban Population'

        df_pop_ceb['Total Urban Population'] = [int(tup[0] * tup[1] * 0.01) for tup in pops_list]
    
        
# Append DataFrame chunk to data: data
        
data = data.append(df_pop_ceb)

    
# Plot urban population data
    
data.plot(kind='scatter', x='Year', y='Total Urban Population')
    plt.show()


# Set the filename: fn
fn = 'ind_pop_data.csv'

# Call plot_pop for country code 'CEB'
plot_pop('ind_pop_data.csv', 'CEB')


# Call plot_pop for country code 'ARB'
plot_pop('ind_pop_data.csv', 'ARB')


###########################

Time Delta, number of days

# Import date
from datetime import date

# Create a date object for May 9th, 2007
start = date(2007, 5, 9)

# Create a date object for December 13th, 2007
end = date(2007, 12, 13)

# Subtract the two dates and print the number of days
print((end - start).days)


###########################

Printing dates in friendly format

# Assign the earliest date to first_date
first_date = min(florida_hurricane_dates)

# Convert to ISO and US formats
iso = "Our earliest hurricane date: " + first_date.isoformat()
us = "Our earliest hurricane date: " + first_date.strftime("%m/%d/%Y")

print("ISO: " + iso)
print("US: " + us)

###########################

Datetime creation

# Import datetime
from datetime import datetime

# Create a datetime object
dt = datetime(2017, 12, 31, 15, 19, 13)

# Replace the year with 1917
dt_old = dt.replace(year=1917)

# Print the results in ISO 8601 format
print(dt_old)


###########################


//average trip time//


# What was the total duration of all trips?


total_elapsed_time = sum(onebike_durations)

# What was the total number of trips?
number_of_trips = len(onebike_durations)
  
# Divide the total duration by the number of trips
print(total_elapsed_time/ number_of_trips)

###########################

Setting timezones


# Create a timezone object corresponding to UTC-4
edt = timezone(timedelta(hours=-4))

# Loop over trips, updating the start and end datetimes to be in UTC-4
for trip in onebike_datetimes[:10]:
  # Update trip['start'] and trip['end']
  trip['start'] = trip['start'].replace(tzinfo=edt)
  trip['end'] = trip['end'].replace(tzinfo=edt)


###########################

unix timezones


# Import datetime
from datetime import datetime

# Starting timestamps
timestamps = [1514665153, 1514664543]

# Datetime objects
dts = []

# Loop
for ts in timestamps:
  dts.append(datetime.fromtimestamp(ts))
  
# Print results
print(dts)





# Shift the index of the end date up one; now subract it from the start date
rides['Time since'] = rides['Start date'] - (rides['End date'].shift(1))

# Move from a timedelta to a number of seconds, which is easier to work with
rides['Time since'] = rides['Time since'].dt.total_seconds()

# Resample to the month
monthly = rides.resample('M', on = 'Start date')

# Print the average hours between rides each month
print(monthly['Time since'].mean()/(60*60))

######################################################

INDEXING AND COLUMN REARRANGEMENT


# Import pandas
import pandas as pd


# Read in filename and set the index: election


election = pd.read_csv(filename, index_col='county')



# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results


results = election[['winner', 'total', 'voters']]



# Print the output of results.head()


print(results.head())


######################################################

SLICING ROWS


# Slice the row labels 'Perry' to 'Potter': p_counties


p_counties = election.loc['Perry':'Potter',:]



# Print the p_counties DataFrame


print(p_counties)



# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev


p_counties_rev = election.loc['Potter':'Perry':-1]



# Print the p_counties_rev DataFrame


print(p_counties_rev)


######################################################


#####Slicing Data frame columns (using .loc method)######
##### Manipulating dataframes with pandas tutorial########


# Slice the columns from the starting column to 'Obama': left_columns


left_columns = election.loc[:,'state': 'Obama']



# Print the output of left_columns.head()


print(left_columns.head())



# Slice the columns from 'Obama' to 'winner': middle_columns


middle_columns = election.loc[:,'Obama':'winner']



# Print the output of middle_columns.head()


print(middle_columns.head())



# Slice the columns from 'Romney' to the end: 'right_columns'


right_columns = election.loc[:,'Romney':]



# Print the output of right_columns.head()


######################################################

#####Filtering Data frames using booleans ######
##### Manipulating dataframes with pandas tutorial########

# Create the boolean array: high_turnout

high_turnout = election['turnout'] > 70


# Filter the election DataFrame with the high_turnout array: high_turnout_df

high_turnout_df = election[high_turnout]


# Print the high_turnout_results DataFrame

print(high_turnout_df)


print(right_columns.head())


######################################################

#####Filtering columns using other columns ######
##### Manipulating dataframes with pandas tutorial########

# Import numpy

import numpy as np


# Create the boolean array: too_close

too_close = election['margin'] < 1


# Assign np.nan to the 'winner' column where the results were too close to call

election.loc[too_close, 'winner'] = np.nan


# Print the output of election.info()

print(election.info())


######################################################

##### FILTERING USING NANs ######
##### Manipulating dataframes with pandas tutorial########

# Select the 'age' and 'cabin' columns: df

df = titanic[['age','cabin']]


# Print the shape of df

print(df.shape)


# Drop rows in df with how='any' and print the shape

print(df.dropna(how='any').shape)


# Drop rows in df with how='all' and print the shape

print(df.dropna(how='all').shape)


# Drop columns in titanic with less than 1000 non-missing values

print(titanic.dropna(thresh = 1000, axis ='columns').info())


######################################################

##### USING VECTORIZED FUNCTIONS ######
##### Manipulating dataframes with pandas tutorial########

# Import zscore from scimpy.stats

from scipy.stats import zscore


# Call zscore with election['turnout'] as input: turnout_zscore

turnout_zscore = zscore(election['turnout'])


# Print the type of turnout_zscore

print(type(turnout_zscore))


# Assign turnout_zscore to a new column: election['turnout_zscore']

election['turnout_zscore'] = turnout_zscore


# Print the output of election.head()

print(election.head())

######################################################

##### USING LOC[] with NONUNIQUE INDEXES######
##### Manipulating dataframes with pandas tutorial########

# Set the index to the column 'state': sales

sales = sales.set_index(['state'])


# Print the sales DataFrame

print(sales)


# Access the data from 'NY'

print(sales.loc[('NY')])


######################################################

##### Indexing multiple levels of a MultiIndex######
##### Manipulating dataframes with pandas tutorial########

# Look up data for NY in month 1: NY_month1

NY_month1 = sales.loc['NY',1]


# Look up data for CA and TX in month 2: CA_TX_month2

CA_TX_month2 = sales.loc[(['CA','TX'],2),:]


# Access the inner month index and look up data for all states in month 2: all_month2

all_month2 = sales.loc[(slice(None),2), ['month', 2],:]

######################################################

#####pivoting all variables######
##### Manipulating dataframes with pandas tutorial########

# Pivot users with signups indexed by weekday and city: signups_pivot

signups_pivot = users.pivot(index='weekday', columns='city', values='signups')


# Print signups_pivot

print(signups_pivot)


# Pivot users pivoted by both signups and visitors: pivot

pivot = users.pivot(index='weekday', columns='city', values=None)


# Print the pivoted DataFrame
print(pivot)

######################################################

#####restoring index order######
##### Manipulating dataframes with pandas tutorial########


# Stack 'city' back into the index of bycity: newusers

newusers = bycity.stack('city')


# Swap the levels of the index of newusers: newusers

newusers = newusers.swaplevel(0, 1)


# Print newusers and verify that the index is not sorted

print(newusers)


# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))

######################################################

##### melting indexes ######
##### Manipulating dataframes with pandas tutorial########


# Reset the index: visitors_by_city_weekday

visitors_by_city_weekday = visitors_by_city_weekday.reset_index()


# Print visitors_by_city_weekday

print(visitors_by_city_weekday)


# Melt visitors_by_city_weekday: visitors

visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')


# Print visitors

print(visitors)

######################################################

##### obtaining  k/v pairs with melt() ######
##### Manipulating dataframes with pandas tutorial########

# Set the new index: users_idx

users_idx = users.set_index(['city', 'weekday'])


# Print the users_idx DataFrame

print(users_idx)


# Obtain the key-value pairs: kv_pairs

kv_pairs = pd.melt(users_idx, col_level=0)


# Print the key-value pairs

print(kv_pairs)

######################################################

##### setting up a pivot table ######
##### Manipulating dataframes with pandas tutorial########

# Create the DataFrame with the appropriate pivot table: by_city_day

by_city_day = users.pivot_table(index='weekday', columns='city')


# Print by_city_day

print(by_city_day)

######################################################

##### using other aggregations in pivot tables ######
##### Manipulating dataframes with pandas tutorial########

# Use a pivot table to display the count of each column: count_by_weekday1

count_by_weekday1 = pd.pivot_table(users, index='weekday', aggfunc='count')


# Print count_by_weekday

print(count_by_weekday1)


# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2

count_by_weekday2 = pd.pivot_table(users, index='weekday', aggfunc=len)


# Verify that the same result is obtained

print('==========================================')

print(count_by_weekday1.equals(count_by_weekday2))

######################################################

##### using margins in pivot tables ######
##### Manipulating dataframes with pandas tutorial########


# Create the DataFrame with the appropriate pivot table: signups_and_visitors

signups_and_visitors = users.pivot_table(index='weekday',aggfunc=sum)


# Print signups_and_visitors

print(signups_and_visitors)


# Add in the margins: signups_and_visitors_total 

signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum, margins=True)

# Print signups_and_visitors_total

print(signups_and_visitors_total)


######################################################

##### Grouping by multiple columns ######
##### Manipulating dataframes with pandas tutorial########


# Group titanic by 'pclass'

by_class = titanic.groupby('pclass')


# Aggregate 'survived' column of by_class by count

count_by_class = by_class.agg('survived').count()


# Print count_by_class
print(count_by_class)


# Group titanic by 'embarked' and 'pclass'

by_mult = titanic.groupby(['embarked','pclass'])


# Aggregate 'survived' column of by_mult by count

count_mult = by_mult.agg('survived').count()


# Print count_mult
print(count_mult)

######################################################

##### Grouping by another series ######
##### Manipulating dataframes with pandas tutorial########


# Read life_fname into a DataFrame: life

life = pd.read_csv(life_fname, index_col='Country')


# Read regions_fname into a DataFrame: regions

regions = pd.read_csv(regions_fname, index_col='Country')


# Group life by regions['region']: life_by_region

life_by_region = life.groupby(regions['region'])


# Print the mean over the '2010' column of life_by_region

print(life_by_region['2010'].mean())

######################################################

##### Computing multiple aggregates of multiple columns ######
##### Manipulating dataframes with pandas tutorial########

The .agg() method can be used with a tuple or list of aggregations as input. When applying multiple aggregations on multiple columns, the aggregated DataFrame has a multi-level column index.
In this exercise, you're going to group passengers on the Titanic by 'pclass' and aggregate the 'age' and 'fare' columns by the functions 'max' and 'median'. You'll then use multi-level selection to find the oldest passenger per class and the median fare price per class.
The DataFrame has been pre-loaded as titanic.

# Group titanic by 'pclass': by_class

by_class = titanic.groupby('pclass')


# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]


# Aggregate by_class_sub by 'max' and 'median': aggregated

aggregated = by_class_sub.agg(['max','median'])


# Print the maximum age in each class

print(aggregated.loc[:, ('age','max')])


# Print the median fare in each class

print(aggregated.loc[:, ('fare','median')])

######################################################

##### Grouping on a function of the index ######
##### Manipulating dataframes with pandas tutorial########

Grouping on a function of the index
Groupby operations can also be performed on transformations of the index values. In the case of a DateTimeIndex, we can extract portions of the datetime over which to group.
In this exercise you'll read in a set of sample sales data from February 2015 and assign the 'Date' column as the index. Your job is to group the sales data by the day of the week and aggregate the sum of the 'Units' column.
Is there a day of the week that is more popular for customers? To find out, you're going to use .strftime('%a') to transform the index datetime values to abbreviated days of the week.
The sales data CSV file is available to you as 'sales.csv'.


# Read file: sales

sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)


# Create a groupby object: by_day

by_day = sales.groupby(sales.index.strftime('%a'))


# Create sum: units_sum

units_sum = by_day['Units'].sum()


# Print units_sum
print(units_sum)

######################################################

##### Filling missing data(imputation) by group ######
##### Manipulating dataframes with pandas tutorial########

# Create a groupby object: by_sex_class

by_sex_class = titanic.groupby(['sex','pclass'])


# Write a function that imputes median

def impute_median(series):
    
return series.fillna(series.median())


# Impute age and assign to titanic.age

titanic.age = by_sex_class.age.transform(impute_median)


# Print the output of titanic.tail(10)

print(titanic.tail(10))