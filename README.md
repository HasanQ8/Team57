# Team57

Code Explanation and Thought Process



`numeric_col =['Year of Record','Size of City','Work Experience in Current Job [years]',
              'Age','Yearly Income in addition to Salary (e.g. Rental Income)'] 

category_col =['Housing Situation','Satisfaction with employer'
,'Gender'
,'Country'
,'Profession'
,'University Degree'] `



Those two showed the value counts in each column in the different categories to see what kind of data we are dealing with. Helped with removing variables likes #NUM! and all type of outliers 

After investigating what column to dropped, we dropped Instance, Wears Glasses, Hair Color, Crime Level and Body Height. which for us resulted in the highest score. 

Year of record and Size of city has the most effect on the score.

Cleaning the data made a little difference in the score. Basically wanted to uniform all missing data for categorical data with one string "Nan_data" and the mean for numerical. 

for year of record looking at the column, all the missing data was after 2019 and since it was sorted the assumption was made that all data missing would be 2019.

Used Target encoder for the categorical features.

Played with the test_size to get the best scores. usually 0.2 and 0.3 made for the best scores.

Used different ensemble models, best performing ones by far were LGB and Cat booster. Cat booster no matter what we did we couldn't get a better score of 12/13k. While lgb always got a score of 11k and below. 



Referred to  https://lightgbm.readthedocs.io/en/latest/Parameters.html and other online resources to try and get the best parameters for lgb. The parameters change will deal with specific problems like a lower num_leaves would deal better with overfitting etc.



Simple code lead to a better score, the more complicated we made things the worse the score got. 



