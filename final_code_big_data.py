import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity




# QUESTION 1

path = r"C:\Users\Avocando\Documents\Deree\MASTERS\Intro to Big Data"

# C:\Users\notis\Documents\Deree\Vogiatzis

# FILE PATHS
ratings_path = path + r'\u.data'
movies_path = path + r'\u.item'
users_path = path + r'\u.user'
info_path = path + r'\u.info'


# LOAD DATASET
ratings = pd.read_csv(ratings_path, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv(movies_path, sep='|', encoding='ISO-8859-1', names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
users = pd.read_csv(users_path, sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

 





user_movie_counts = ratings['user_id'].value_counts()

# Plot 1
plt.figure(figsize=(10, 6))
user_movie_counts.plot(kind='hist', bins=50)
plt.xlabel('Number of Movies Seen')
plt.ylabel('Number of Users')
plt.title('Distribution of Movies Seen by Each User')
plt.show()



rating_counts = ratings['rating'].value_counts().sort_index()

# Plot 2
plt.figure(figsize=(8, 5))
rating_counts.plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Frequency of Each Rating')
plt.show()


movie_counts = ratings['item_id'].value_counts()


user_z_scores = zscore(user_movie_counts)
movie_z_scores = zscore(movie_counts)

user_outliers = user_movie_counts[user_z_scores > 3]
movie_outliers = movie_counts[movie_z_scores > 3]

# Display the outliers
print("User Outliers")


print("================")

print(user_outliers)
print("\n")

print("Movie Outliers")

print("================")

print(movie_outliers)

pivot1 = pd.pivot_table(ratings, values= ["rating"] , index = ["user_id"] , columns=["item_id"]) 


print(pivot1)

# QUESTION 2

ratings_test = path + r'\u1.test'
ratings_base = path + r'\u1.base'

# LOAD DATASET

#ratings_test1 = pd.read_csv(ratings_test, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
#ratings_base1 = pd.read_csv(ratings_base, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# sample 80% of the data in the results dataframe, give it to ratings_base1 and give the other 20% to ratings_test1.
# the parameter random_state is to be sure that the results are the same every time we run the code (reproducibility)

ratings_base1 = ratings.sample(frac = 0.8, random_state = 40)
ratings_test1 = ratings.drop(ratings_base1.index)

# FIND THE MOVIES WITH THE BEST AVERAGE RATING



top_movies = ratings_base1.groupby("item_id")["rating"].mean().sort_values(ascending=False)


# movie_review_counts counts how many reviews each movie has received

movie_review_counts = ratings_base1.groupby("item_id").size().reset_index(name="review_count")

# median_count finds the median number of reviews of each movie

#median_count = movie_review_counts.median()

median_reviews_count = movie_review_counts["review_count"].median()
movie_review_average = ratings_base1.groupby("item_id").mean()



# ***********************************************************
#                  FORMULA FOR WEIGHTED RATING (W)
# 
# R : Mean of each movie (item-specific)
# V : Number of ratings for each movie (item-specific)
# C : Mean of ratings of all movies (universal)
# M : Minimum number of votes for a movie to be considered 
#     amongst the top (universal)
#
# Weighted Rating (WR) Formula:
# WR = (V ÷ (V + M)) × R + (M ÷ (V + M)) × C
# ***********************************************************


# M is the minimum number of movies that we decided is the 69% of the median in order to be considered in the top movies list

median_percentage = 0.69 
M = median_reviews_count * median_percentage


C = ratings_base1['rating'].mean()



R = movie_review_average["rating"]

V = ratings_base1.groupby("item_id")["rating"].count()



# WR formula implementation
W = (V / (V + M)) * R + (M / (V + M)) * C




# Function to map numeric ratings to verbal ratings
def verbal_rating(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Positive"
    elif rating >= 4:
        return "Very Positive"

top_movies_by_WR = W.sort_values(ascending=False)

# TOP 10 RATED MOVIES ACCORDING TO W
top_10_movies = top_movies_by_WR.head(10)

# Display the top 10 movies
print("Top 10 Movies by Weighted Rating:")
print(top_10_movies)




top_10_movies_with_titles = pd.DataFrame({'item_id': top_10_movies.index, 'Weighted Rating': top_10_movies.values})
top_10_movies_with_titles = top_10_movies_with_titles.merge(movies[['movie_id', 'movie_title']], 
                                                            left_on='item_id', 
                                                            right_on='movie_id')[['movie_title', 'Weighted Rating']]


top_10_movies_with_titles.columns = ['Movie Title', 'Weighted Rating']



top_10_movies_with_titles.index = pd.RangeIndex(start=1, stop=(len(top_10_movies_with_titles) + 1))


# Display the top 10 movies with titles
print("Top 10 Movies by Weighted Rating:")
print(top_10_movies_with_titles)

#  10 RANDOM MOVIES

# It shows 10 random movies based on the rankings given by the database NOT WR!
random_recommendations = ratings_base1[['item_id', 'rating']].sample(10)

# Merge 10 random movies with titles
random_recommendations_with_titles = random_recommendations.merge(movies[['movie_id', 'movie_title']], 
                                                                  left_on='item_id', 
                                                                  right_on='movie_id')[['movie_title', 'rating']]


random_recommendations_with_titles['Verbal Rating'] = random_recommendations_with_titles['rating'].apply(verbal_rating)

# Rename columns for clarity
random_recommendations_with_titles.columns = ['Movie Title', 'Rating', 'Verbal Rating']

random_recommendations_with_titles.index = pd.RangeIndex(start=1, stop=(len(random_recommendations_with_titles) + 1))


# Display the 10 random movies with titles
print("\n10 Random Movies:")
print(random_recommendations_with_titles)

#====================================================================================================



# **********************************************************
#                   WHAT I DID:
#       
#
# 1. The instructor wanted for us to find the precision and recall for each user as i understood.
# Then he wanted  to add all those scores and calculate an average. Those scores were calculated by seeing for each users the movies the best (!) movies that they like 
# and the recommendations that we did. Then we can calculate the total F1 score by using the averages. 
#
#2. The F1 score here seems really low. I think that the problem is that when i try to find the favorite movies of a specific user i just find the top 5 rates of them.
# and the formula always searches from the begining to find the top rated ones.  
#
#************************************************************

############ FINDING THE F1 SCORE FOR SYSTEM RECOMMENDATIONS


# FIND THE AVERAGE PRECISION AND RECALL OF ALL THE TEST USERS. FIRST INITIALIZE SOME VARIABLES

# top_5_movies show the best movies according the W score that we calculated above
top_5_movies = top_movies_by_WR.head(5)

# we initialize the recall_general as a sum of all user recall scores. Afterwards we will divide it by the number of users
recall_general_q2_recommender = 0 
# We do the same for precision_genral for the precision score
precision_general_q2_recommeder = 0
 
# system_recommendation is  a list of the top 5 movies of the system
system_recommendation = list(top_5_movies.index)

# rate_by_user is a pivot that shows the ratings of all movies by user. Each row is a specific user
rate_by_user = pd.pivot_table(ratings_base1, values= ["rating"] , index = ["user_id"] , columns=["item_id"])

# ITERATE BY USER

# we start by iterating the rate_by_user row by row (that is a user by user iteration)
for user, ratings in rate_by_user.iterrows():
    
    # FIND THE RACALL OF EACH USER. IT IS A COMPARISON BETWEEN THE ACTUAL PREFERENCES AND THE SESTEM RECOMMENDATIONS 
    
    # we initialize recall_user as a count. We will add +1 every time a movie of actual preference of a user is included in the system recommendation movies
    recall_user = 0
 
 
    # From each user we find the 5 best rated movies and we save them in best_5_actual_ratings
    best_5_actual_ratings = ratings.nlargest(5).index
    
    # list_of_actual is a list of the 5 best movies of a single user. We just make the previous variable as a list for easier use.    
    list_of_actual = []
    for rating, movie in best_5_actual_ratings:
        list_of_actual.append(movie)

    # iterate the list_of_actual to calculate the recall parameter for each user    
    for movie in list_of_actual:
        # if an actually preferred movie of a user is in our system recommended users, than add 1 torecall_user
        if movie in system_recommendation:
            recall_user += 1
    
    # Now divide the recall user by 5 (number of actually prefferd movies by the user) to find the true recall of that user
    recall_user = recall_user/5
    
    # print(f"The recall of user: " + str(user) + " is: " + str(recall_user))
 
    # ADD THE USER SPECIFIC RECALL TO THE GENERAL RECALL
    
    # Now add the recall of the specific user to the recall_general
    recall_general_q2_recommender += recall_user
    
    
    # FIND THE PRECISION OF EACH USER. IT IS A COMPARISON BETWEEN THE ACTUAL PREFERENCES AND THE SESTEM RECOMMENDATION
    
    # Do the same with precision. Precision user is a count initialized by 0. Whenever a movie from our system preference movie list is actually a favorite movie ogf the user then add 1 to the count.  
    precision_user = 0
    
    for recommended in system_recommendation:
        
        if recommended in list_of_actual:
            precision_user += 1
    
    # Divide the precision by the number of system recommended movies
    precision_user = precision_user / len(system_recommendation)

    # print(f"The precision of user: " + str(user) + " is: " + str(precision_user))

    # ADD THE USER SPECIFIC PRECISION TO THE GENERAL PRECISION
    
    # Now add the precision_user to the precison general
    precision_general_q2_recommeder += precision_user


# FIND THE RECALL AND PRECISION BY DIVIDING BY THE SUM OF USERS

recall_general_q2_recommender = recall_general_q2_recommender/len(rate_by_user)
    
precision_general_q2_recommeder = precision_general_q2_recommeder/len(rate_by_user)

# CALCULATE THE FORMULA OF F1    
F1_q2_recommender = 2 * recall_general_q2_recommender * precision_general_q2_recommeder /(recall_general_q2_recommender + precision_general_q2_recommeder)
    
# For the version 2 (random movies) whenever we used the system_recommendation variable we will now use the method sample to create a list of 5 random movies. 

################################### DO THE SAME FOR RANDOM RECCOMENDATIONS ##################################################


################################################################################################


# FIND THE AVERAGE PRECISION AND RECALL OF ALL THE TEST USERS. FIRST INITIALIZE SOME VARIABLES


# we initialize the recall_general as a sum of all user recall scores. Afterwards we will divide it by the number of users
recall_general_q2_random = 0 
# We do the same for precision_genral for the precision score
precision_general_q2_random = 0
 

# ITERATE BY USER

# we start by iterating the rate_by_user row by row (that is a user by user iteration)
for user, ratings in rate_by_user.iterrows():
    
    # FIND THE RACALL OF EACH USER. IT IS A COMPARISON BETWEEN THE ACTUAL PREFERENCES AND THE RANDOM RECOMMENDATIONS 
    
    # we initialize recall_user as a count. We will add +1 every time a movie of actual preference of a user is included in the random recommendation movies
    recall_user = 0
 
    

 
    # From each user we find the 5 best rated movies and we save them in best_5_actual_ratings
    best_5_actual_ratings = ratings.nlargest(5).index
    
    # list_of_actual is a list of the 5 best movies of a single user. We just make the previous variable as a list for easier use.    
    list_of_actual = []
    for rating, movie in best_5_actual_ratings:
        list_of_actual.append(movie)

    # iterate the list_of_actual to calculate the recall parameter for each user    
    for movie in list_of_actual:
        # if an actually preferred movie of a user is in our system recommended users, than add 1 torecall_user
        if movie in system_recommendation:
            recall_user += 1
    
    # Now divide the recall user by 5 (number of actually prefferd movies by the user) to find the true recall of that user
    recall_user = recall_user/5
    
    # print(f"The recall of user: " + str(user) + " is: " + str(recall_user))
 
    # ADD THE USER SPECIFIC RECALL TO THE GENERAL RECALL
    
    # Now add the recall of the specific user to the recall_general
    recall_general_q2_random += recall_user
    
    
    # FIND THE PRECISION OF EACH USER. IT IS A COMPARISON BETWEEN THE ACTUAL PREFERENCES AND THE RANDOM RECOMMENDATION

    # Create for every user a list of five random recommendations
    random_recommedations = ratings_base1[['item_id', 'rating']].sample(5)
    
    # create a list out of it with only the movies 
    random_list = list(random_recommedations["item_id"].values)
    
    # Do the same with precision. Precision user is a count initialized by 0. Whenever a movie from our system preference movie list is actually a favorite movie ogf the user then add 1 to the count.  
    precision_user = 0
    
    for recommended in random_list:
        
        if recommended in list_of_actual:
            precision_user += 1
    
    # Divide the precision by the number of system recommended movies
    precision_user = precision_user / len(system_recommendation)

    # print(f"The precision of user: " + str(user) + " is: " + str(precision_user))

    # ADD THE USER SPECIFIC PRECISION TO THE GENERAL PRECISION
    
    # Now add the precision_user to the precison general
    precision_general_q2_random += precision_user


# FIND THE RECALL AND PRECISION BY DIVIDING BY THE SUM OF USERS

recall_general_q2_random = recall_general_q2_random/len(rate_by_user)
    
precision_general_q2_random = precision_general_q2_random/len(rate_by_user)

# CALCULATE THE FORMULA OF F1    
F1_q2_random = 2 * recall_general_q2_random * precision_general_q2_random /(recall_general_q2_random + precision_general_q2_random)

# SO FOR Q2 in general we have two F1's
print("We have the F1 score from our system recommendations: {}".format(F1_q2_recommender))
print("And we also have the F1 score from our random recommendations per user: {}".format(F1_q2_random))

#====================================================================================================


#Q3
#create a pivot table that has users as rows,movies as columns and ratings as values
pivot2 = pivot1.fillna(0)
#convert the table into numpy array for similarity calculations
r = np.array(pivot2)





nUsers=r.shape[0]
nItems=r.shape[1]
#Find k-most similar users using cosine similarity




def findKSimilar(r, k):
    # similarUsers is 2-D matrix
    similarUsers = -1 * np.ones((nUsers, k))
    similarities = cosine_similarity(r)
    
    # loop for each user
    for i in range(0, nUsers):
        simUsersIdxs = np.argsort(similarities[:, i])[::-1]  # Sort the similarities of user with the others in descending order
        similarUsers[i, :] = simUsersIdxs[1:k+1]  # Assign the k-most similar users to user, excluding themselves
        
            
    return similarUsers, similarities







def predict(userId, itemId, r,similarUsers,similarities):

    nCols=similarUsers.shape[1]  # number of similar users want to consider
    sum=0.0 # Weighted sum of ratings
    simSum=0.0 #total sum of similarities
    # Loop over the k most similar users
    for l in range(0,nCols):    
        neighbor=int(similarUsers[userId, l]) # Get the index of the l-th most similar user from the current user
        # If the neighbor has rated the item, use their rating to contribute to the weighted sum
        sum= sum+ r[neighbor,itemId]*similarities[neighbor,userId] 
        # Add the similarity between the current user and the neighbor to the similarity sum
        simSum = simSum + similarities[neighbor,userId] 
        
    
    if simSum > 0: # check for similarity
        return sum / simSum  # Return weighted average of ratings
    else:
        return 0  # if there is no similarity
    
# Split data: hide 20% of the ratings
def hide_data(r, percentage=0.2, seed=42): # to ensure that the hidden cells will stay the same every time we run the code
    np.random.seed(seed)  # Set random seed for reproducibility totproduce the same results every time we run the code & hide the same set of cells
    mask = np.random.rand(*r.shape) < percentage  # Randomly mask cells
    r_hidden = r.copy() # to ensure that the original matrix won't modified
    test_indices = np.where(mask)  # Ensure to track the indices of hidden ratings
    r_hidden[test_indices] = 0  # Set hidden cells to 0, so the model cannot access them during training or prediction
    return r_hidden, test_indices

# Evaluate only on hidden cells
def evaluate_hidden_cells(r, predictions, test_indices):
    mae=0 #initialize the evaluation measures
    rmse=0 
    tp=0 
    fp=0 
    fn=0
    tn =0
    # loop through the indices of hidden cells and use zip() to iterate over both the rows (users) and columns (items) at the same time, to evaluate the hidden cells.
    for userId, itemId in zip(test_indices[0], test_indices[1]):
        actual = r[userId, itemId]  # retrieve the true rating
        predicted = predictions[userId, itemId]  # retrieve the predicted rating
        mae += abs(predicted - actual)
        rmse += (predicted - actual) ** 2
        
        # For Precision/Recall/F1
        if predicted >= 2 and actual >= 2:
            tp += 1
        elif predicted >= 2 and actual < 2:
            fp += 1
        elif predicted < 2 and actual >= 2:
            fn += 1
        elif predicted < 2 and actual < 2:
            tn += 1
            
    n = len(test_indices[0])  #Calculate the number of hidden cells, based on the test_indices
    mae /= n
    rmse = np.sqrt(rmse / n)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return mae, rmse, precision, recall, f1

# Generate predictions for all cells in the matrix
def generate_predictions(r, similarUsers, similarities):
    nUsers, nItems = r.shape #get the number of users and items
    predictions = np.zeros((nUsers, nItems)) # Initialize a matrix to hold the predicted ratings
    for userId in range(nUsers):
        for itemId in range(nItems):
            predictions[userId, itemId] = predict(userId, itemId, r, similarUsers, similarities)  # Predict the rating for the current user-item pair 
    return predictions

def generate_top5_recommendations(predictions, r_hidden, k=5):
    top5_recommendations = {} 
    # For each user, get the top 5 items based on predicted ratings
    for userId in range(predictions.shape[0]):
        # Get the predicted ratings for the current user
        user_predictions = predictions[userId, :]       
        # Mask out the items that the user has already rated
        # We assume that non-zero values in r_hidden are already rated
        user_predictions[r_hidden[userId, :] != 0] = -np.inf  # Mask already rated items by setting them to a very low value       
        # Get the indices of the top 5 highest predicted ratings
        top5_indices = np.argsort(user_predictions)[::-1][:k]  # Sort and take the top 5 indices       
        # Store the top 5 recommendations for the current user
        top5_recommendations[userId] = top5_indices
    
    return top5_recommendations








# Hide 20% of the data
r_hidden, test_indices = hide_data(r, percentage=0.2)



# Calculate similar users and their similarities                
similarUsers, similarities=findKSimilar (r,2)





# Generate predictions for all cells
predictions = generate_predictions(r_hidden, similarUsers, similarities)


# Evaluate on hidden cells
mae, rmse, precision, recall, f1 = evaluate_hidden_cells(r, predictions, test_indices)
# Print results
print("MAE:", mae)
print("RMSE:", rmse)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)




    
# Generate top 5 recommendations for each user
top5_recommendations = generate_top5_recommendations(predictions, r_hidden, k=5)

# Print the top 5 recommendations for each user
# for userId, recommendations in top5_recommendations.items():
#     print(f"User {userId} top 5 recommendations: {recommendations}")

# ALL RESULTS FUNCTIONS

#Function to show Question 2 results
def show_q2_results():
    print("\n===== Question 2 Results =====")
    
    # Display the top 10 movies with titles
    print("Top 10 Movies by Weighted Rating:")
    print(top_10_movies_with_titles)
    
    
    # Display the 10 random movies with titles
    print("\n10 Random Movies:")
    print(random_recommendations_with_titles)
    
    
    print(f"F1 Score from System Recommendations: {F1_q2_recommender:.4f}")
    print(f"F1 Score from Random Recommendations: {F1_q2_random:.4f}")

# Function to show Top 5 Recommendations for each user

def show_recommendations():
    print("\nTop 5 Recommendations for Each User:")
    for userId, recommendations in top5_recommendations.items():
        print(f"User {userId} top 5 recommendations: {recommendations}")
        
        
        
        
        
 #====================================================================================================
 
 
 
 # Q4
       
        
def calculate_age_similarity(users, age_weight=1):
    """
    Calculate the age similarity matrix based on age differences.
    """
    n_users = len(users)
    age_similarity = np.zeros((n_users, n_users))
    
    # Loop through each pair of users to compute age similarity
    for i in range(n_users):
        for j in range(n_users):
            age_diff = abs(users.iloc[i]['age'] - users.iloc[j]['age'])
            # Similarity decreases as age difference increases
            age_similarity[i, j] = 1 / (1 + age_diff)
    
    return age_similarity

import time

def find_enhanced_similar_users(r, k, users, w1=0.5, w2=0.5):
    """
    Find the enhanced k-most similar users by combining cosine similarity and age similarity.
    """
    n_users = r.shape[0]
    enhanced_similar_users = -1 * np.ones((n_users, k))

    # Cosine similarity matrix
    user_similarity = cosine_similarity(r)

    # Age similarity matrix
    age_similarity = calculate_age_similarity(users)

    # Combine similarities
    enhanced_similarity = (w1 * user_similarity + w2 * age_similarity) / (w1 + w2)

    # Loop through each user to find k-most similar users
    for i in range(n_users):
        sim_users_idxs = np.argsort(enhanced_similarity[:, i])[::-1]
        enhanced_similar_users[i, :] = sim_users_idxs[1:k+1]

    return enhanced_similar_users, enhanced_similarity

def predict_enhanced(userId, itemId, r, enhanced_similar_users, enhanced_similarity):
    """
    Predict ratings based on enhanced user similarity.
    """
    nCols = enhanced_similar_users.shape[1]
    weighted_sum = 0.0
    sim_sum = 0.0

    for l in range(nCols):
        neighbor = int(enhanced_similar_users[userId, l])
        if r[neighbor, itemId] != 0:
            weighted_sum += r[neighbor, itemId] * enhanced_similarity[neighbor, userId]
            sim_sum += enhanced_similarity[neighbor, userId]

    return weighted_sum / sim_sum if sim_sum > 0 else 0

# Generate predictions using enhanced similarity and evaluate for multiple weight pairs
weight_pairs = [(0.7, 0.3), (0.6, 0.4), (0.8, 0.2)]

print("\n===== Question 4 Results for Different Weight Combinations =====")

for w1, w2 in weight_pairs:
    print(f"\nWeights: w1 = {w1} , w2 = {w2}")

    print("Calculating... Please wait.")
    

    # Calculate enhanced similar users
    enhanced_similar_users, enhanced_similarity = find_enhanced_similar_users(r, k=2, users=users, w1=w1, w2=w2)

    enhanced_predictions = generate_predictions(r_hidden, enhanced_similar_users, enhanced_similarity)

    mae_enhanced, rmse_enhanced, precision_enhanced, recall_enhanced, f1_enhanced = evaluate_hidden_cells(r, enhanced_predictions, test_indices)
    
    print("\n")
    print("+-----------------------------------+")
    print(f"| Enhanced MAE: {mae_enhanced:.4f}         ")
    print(f"| Enhanced RMSE: {rmse_enhanced:.4f}       ")
    print(f"| Enhanced Precision: {precision_enhanced:.4f}  ")
    print(f"| Enhanced Recall: {recall_enhanced:.4f}     ")
    print(f"| Enhanced F1 Score: {f1_enhanced:.4f}     ")
    print("+-----------------------------------+")








