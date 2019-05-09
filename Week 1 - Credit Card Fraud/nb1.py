#%%
from sklearn.tree._criterion import ClassificationCriterion

class SqrtGini(ClassificationCriterion):
    

#%%
t_day = df.groupby('creationdate')['amount'].agg(['count'])

plt.figure(1, figsize=(15,7))
fig = plt.gcf()
ax = plt.gca()

t_day.plot(ax=ax)
ax.set(xlabel="Date", ylabel="Number of transactions")
ax.xaxis.set_major_locator(mdates.WeekdayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.get_legend().remove()


#%%
t_ip = df.groupby('ip_id')['amount'].agg(['count'])
t_ip = t_ip.sort_index()

plt.figure(2, figsize=(15,7))
fig = plt.gcf()
ax = plt.gca() 

t_ip.plot(ax=ax)
ax.set(xlabel="IP", ylabel="Number of transactions")
ax.get_legend().remove()


#%%
preprocessed = df.drop(['label', 'creationdate_stamp', 'creationdate'], axis=1)

# clf = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(clf, preprocessed, df['label'], cv=3)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#%%
# If we want to use a all-combiner. This should yield less false positives. 
# from brew.base import Ensemble
# from brew.base import EnsembleClassifier
# from brew.combination.combiner import Combiner

# # create your Ensemble
# clfs = your_list_of_classifiers # [clf1, clf2]
# ens = Ensemble(classifiers = clfs)

# # create your Combiner
# # the rules can be 'majority_vote', 'max', 'min', 'mean' or 'median'
# comb = Combiner(rule='mean')

# # now create your ensemble classifier
# ensemble_clf = EnsembleClassifier(ensemble=ens, combiner=comb)
# ensemble_clf.predict(X)
