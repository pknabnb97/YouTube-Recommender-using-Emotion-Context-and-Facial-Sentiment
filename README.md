# YouTube-Recommender-using-Emotion-Context-and-Facial-Sentiment

This is a simple recommender that uses YouTube-Api to recommend videos.
The app first takes in Emotion and keeps it as context. If recommended video gets negative sentiment feedback, 
new category of video is recommended, keeping the Emotion it encountered first as context for the next recommendation.

*Only the highest emotion/sentiment is taken, but the results are stored for all emotion. 
*These stored information can be used as features for an intelligent Recommender System.
*These can also be recorded for the whole-day/work-hours to give intelligent mood-based Recommendation
Although this Recommender is simple mapping, it can be made intelligent by implementing personalised ranking to the categories
with successive usage. Collaborative Filtering would make it more efficient. 
*But this will require crowdsourcing and collecting facial data for every video watched

YOUTUBE VIDEO_CATEGORY_ID LIST CAN BE FOUND ON THE GITHUB DIRECTORY.
    
#=========CONTEXT======================MAPPED-TO========================#
#=========EMOTION==================VIDEO_CATEGORY_ID====================#
#==========SAD======================23, 15, 10==========================#
#==========HAPPY====================24, 40, 28==========================#
#==========ANGRY/STRESSED===========24, 39, 15==========================#
#==========SURPRISE/FEAR============1, 10, 39===========================#

SURPRISE and NEUTRAL are not good features for this first part of the recommender,
but SURPRISE is useful in SENTIMENT FEEDBACK.
FOR SENTIMENT FEEDBACK: 
If user is SAD/HAPPY/SURPRISE mostly while watching the video, it means user is engaged with
the recommended media. So a POSITIVE SENTIMENT is taken.
                         
ANGRY and DISGUST generally gives misclassification error in most FER tests,
as they look almost the same. In this recommendation, we can treat both anger and disgust
as NEGATIVE SENTIMENT as Anger is bad for health and very few people like Disgusting videos.
                         
To Train the model yourself:
1. Download FER+ dataset through GitHub.
2. Extract the images of the original FER2013 dataset.
3. Place all Train, Test and Validation images into the single folder in "current_directory/dataset/images"
4. Run this model.py file.

FER+ has less misclassification error than FER2013 as there is no incorrect labelling.
The state-of-the-art techniques used in training FER2013 can be implemented here to increase accuracy.


