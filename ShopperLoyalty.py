
# coding: utf-8

# # Prediction of Shopper Loyalty 
# ### Solution to Kaggle's Acquiring Valued Shoppers Challenge
# *K Madhumathi, Tarun Tater, Tanmayee Narendra*
# 
# Challenge description -- https://www.kaggle.com/c/acquire-valued-shoppers-challenge
# 
# Data files may be found here -- https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
# 
# For more details about solution, please refer ml_project_report.pdf

# ### Code
# All import Statements 

# In[ ]:

import graphlab as gl
import re
import time
from datetime import datetime
import graphlab.aggregate as agg
from graphlab import SArray


# Reading different files

# In[ ]:

trainHistory = gl.SFrame.read_csv("trainHistory.csv")
transactions = gl.SFrame.read_csv("transactions.csv")
offerData = gl.SFrame.read_csv("offers.csv")
trainFeatures = trainHistory.join(offerData, on='offer', how='left') #joining the offer file and trainHistory


# Creating the 'items.csv' file which has products belonging to only those categories or brand or company which have offer on them

# In[ ]:

from datetime import datetime
loc_offers = "offers.csv"
loc_transactions = "transactions.csv"
loc_reduced = "items.csv" # will be created

def reduce_data(loc_offers, loc_transactions, loc_reduced):

    start = datetime.now()  
    brands = {}
    categories = {}
    companies = {}
    
    for e, line in enumerate( open(loc_offers) ):
        a = line.split(",")[5]
        a = re.sub("""\n""", "", a, re.I|re.S)
        brands[ a ] = 1
        categories[ line.split(",")[1] ] = 1
        companies[ line.split(",")[3] ] = 1
        
  #open output file
    with open(loc_reduced, "wb") as outfile:
    #go through transactions file and reduce
        for e, line in enumerate( open(loc_transactions) ):
            if e == 0:
                outfile.write( line ) #print header
            else:
        #only write when category in offers dict
                if ((line.split(",")[3] in categories)|(line.split(",")[4] in companies)|(line.split(",")[5] in brands)):
                    outfile.write( line )
      #progress
            if e % 5000000 == 0:
                print e, datetime.now() - start
    print e, datetime.now() - start

reduce_data(loc_offers, loc_transactions, loc_reduced)


# reading items file in a sframe

# In[ ]:

itemsBrand = items[:]
itemsCategory = items[:]
itemsCompany = items[:]
itemsBrand.remove_columns(['chain', 'dept', 'date', 'productsize','productmeasure','category','company'])
itemsCategory.remove_columns(['chain', 'dept', 'date', 'productsize','productmeasure','brand','company'])
itemsCompany.remove_columns(['chain', 'dept', 'date', 'productsize','productmeasure','category','brand'])


# In[ ]:




# Storing the start and end indices of each user in the items file 

# In[ ]:

tic = time.time()
listofIndices = {}
lId = []
prevId = "86246"
e1 = 0
l = [e1]
lStart = []
with open("items.csv") as f:
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        if(prevId == line.split(",")[0]):
            pass
        else:
            l.append(e)
            listofIndices[prevId] = l
            lId.append(prevId)
            prevId = line.split(",")[0]
            e1 = e
            l = [e1]
            lStart.append(int(e1))
            
l = [e1, e]
listofIndices[line.split(",")[0]] = l
toc = time.time()
print (toc - tic)
print len(listofIndices)


# Extracting Features

# In[ ]:

trainFeatures = gl.SFrame.read_csv("trainHistory.csv")
trainFeatures['repeater'] = (trainFeatures['repeater'] == 'f')
testFeatures = gl.SFrame.read_csv('testHistory.csv')
offerData = gl.SFrame.read_csv("offers.csv")
trainFeatures = trainFeatures.join(offerData, on='offer', how='left')


# In[ ]:

*************************************


# In[ ]:

#Recording brand, category and company for each row in train data
tic = time.time()
dictStore = {}
lIdT = []
l = []
with open("trainFeatures.csv") as f:
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        l = [int(line.split(",")[7]), int(line.split(",")[9]), int(line.split(",")[11])]
        dictStore[int(line.split(",")[0])] = l
        lIdT.append(int(line.split(",")[0]))
        toc = time.time()
print (toc - tic)
print len(dictStore)
print len(lIdT)

tic = time.time()
dictStoreTest = {}
lIdTTest = []
lTest = []
with open("testFeatures.csv") as f:
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        l = [int(line.split(",")[5]), int(line.split(",")[7]), int(line.split(",")[9])]
        dictStoreTest[int(line.split(",")[0])] = l
        lIdTTest.append(int(line.split(",")[0]))
        toc = time.time()
print (toc - tic)
print len(dictStoreTest)
print len(lIdTTest)


# In[ ]:

*************************************


# In[ ]:

#9 features : 
tic = time.time()
i = 0
listOfValuesBrand = []
listOfValuesCompany = []
listOfValuesCategory = []
keyError = []
for x in lIdT:
    i = i + 1
    presentValues = [0,0,0]
    if((i % 400) == 0):
        print (time.time()-tic), i
    try:
        xlist = listofIndices[x]
        particularIdBrand = itemsBrand[xlist[0]:xlist[1]]#particular id is a sframe
        particularIdCompany = itemsCompany[xlist[0]:xlist[1]]
        particularIdCategory = itemsCategory[xlist[0]:xlist[1]]

        listDictStore = dictStore[x]
        particularIdCategory = particularIdCategory.filter_by([listDictStore[0]], 'category')    
        if(len(particularIdCategory)!= 0):
            presentValues = [len(particularIdCategory), sum(particularIdCategory['purchasequantity']), sum(particularIdCategory['purchaseamount'])]

        listOfValuesCategory.append(presentValues)

        particularIdCompany = particularIdCompany.filter_by([listDictStore[1]], 'company')    
        if(len(particularIdCompany)!= 0):
            presentValues = [len(particularIdCompany), sum(particularIdCompany['purchasequantity']), sum(particularIdCompany['purchaseamount'])]
        listOfValuesCompany.append(presentValues)
        
        particularIdBrand = particularIdBrand.filter_by([listDictStore[2]], 'brand')    
        if(len(particularIdBrand)!= 0):
            presentValues = [len(particularIdBrand), sum(particularIdBrand['purchasequantity']), sum(particularIdBrand['purchaseamount'])]
        listOfValuesBrand.append(presentValues)
    
    except:
        keyError.append(x)
        print x

toc = time.time()
print (toc-tic)

presentValues=[0,0,0]
for x in keyError:
    position = lIdT.index(x)
    listOfValuesBrand.insert(position, presentValues)
    listOfValuesCategory.insert(position, presentValues)
    listOfValuesCompany.insert(position, presentValues)
    
numCategoryTest = []
quanCategoryTest = []
amountCategoryTest = []
for x in listOfValuesCategoryTest:
    numCategoryTest.append(x[0])
    quanCategoryTest.append(x[1])
    amountCategoryTest.append(x[2])
    
numCompanyTest = []
quanCompanyTest = []
amountCompanyTest = []
for x in listOfValuesCompanyTest:
    numCompanyTest.append(x[0])
    quanCompanyTest.append(x[1])
    amountCompanyTest.append(x[2])
    
numBrandTest = []
quanBrandTest = []
amountBrandTest = []
for x in listOfValuesBrandTest:
    numBrandTest.append(x[0])
    quanBrandTest.append(x[1])
    amountBrandTest.append(x[2])


# In[ ]:

*************************************


# Adding individual Feature to sframes

# In[ ]:

training = gl.SFrame()
userId = SArray(lIdT)
training.add_column(userId)
Testing = gl.SFrame()
userId = SArray(lIdTTest)
Testing.add_column(userId)

numCompany = SArray(numCompany)
training.add_column(numCompany)
numCategory = SArray(numCategory)
training.add_column(numCategory)
numBrand = SArray(numBrand)
training.add_column(numBrand)

numCompanyTest = SArray(numCompanyTest)
Testing.add_column(numCompanyTest)
numCategoryTest = SArray(numCategoryTest)
Testing.add_column(numCategoryTest)
numBrandTest = SArray(numBrandTest)
Testing.add_column(numBrandTest)

numBrandTest = SArray(numBrandTest)
Testing.add_column(numBrandTest)

Testing.rename({'X1' : 'id', 'X2' : 'numCompany', 'X3' : 'numCategory', 'X4' : 'numBrand'})

quanCompany = SArray(quanCompany)
training.add_column(quanCompany)
quanCategory = SArray(quanCategory)
training.add_column(quanCategory)
quanBrand = SArray(quanBrand)
training.add_column(quanBrand)

quanCompanyTest = SArray(quanCompanyTest)
Testing.add_column(quanCompanyTest)
quanCategoryTest = SArray(quanCategoryTest)
Testing.add_column(quanCategoryTest)
quanBrandTest = SArray(quanBrandTest)
Testing.add_column(quanBrandTest)

Testing.rename({'X5' : 'quanCompany', 'X6' : 'quanCategory', 'X7' : 'quanBrand'})

amountCompany = SArray(amountCompany)
training.add_column(amountCompany)
amountCategory = SArray(amountCategory)
training.add_column(amountCategory)
amountBrand = SArray(amountBrand)
training.add_column(amountBrand)

amountCompanyTest = SArray(amountCompanyTest)
Testing.add_column(amountCompanyTest)
amountCategoryTest = SArray(amountCategoryTest)
Testing.add_column(amountCategoryTest)
amountBrandTest = SArray(amountBrandTest)
Testing.add_column(amountBrandTest)

Testing.rename({'X8' : 'amountCompany', 'X9' : 'amountCategory', 'X10' : 'amountBrand'})


# In[ ]:

****************************************


# Saving the sframes and loading back.

# In[ ]:

testing.save('./testing')
Testing.save('./testing')

training = gl.load_sframe('./training')
testing = gl.load_sframe('./Testing')
trainHistory = gl.SFrame.read_csv("trainHistory.csv")
offerData = gl.SFrame.read_csv("offers.csv")
trainFeatures = trainHistory.join(offerData, on='offer', how='left')


# ******************************************
# mainpulating sframe to have features as strings and integers as required

# In[ ]:

trainFeatures['repeater'] = (trainFeatures['repeater'] == 'f')
trainFeatures.remove_column('repeattrips')
training1 = training[:]
training.remove_column('id')
trainFeatures.add_columns(training)
temp = trainFeatures[:]
temp.remove_columns(['id','brand', 'company', 'category', 'offer', 'chain', 'market'])
trainFeaturesUpdated = gl.SFrame()
trainFeaturesUpdated.add_columns([trainFeatures['id'].astype(str), trainFeatures['brand'].astype(str), trainFeatures['company'].astype(str), trainFeatures['category'].astype(str), trainFeatures['offer'].astype(str), trainFeatures['chain'].astype(str), trainFeatures['market'].astype(str)], ['id','brand', 'company', 'category', 'offer', 'chain', 'market'])

temp = testFeatures[:]
temp.remove_columns(['id','brand', 'company', 'category', 'offer', 'chain', 'market'])
testFeaturesUpdated = gl.SFrame()
testFeaturesUpdated.add_columns([testFeatures['id'].astype(str), testFeatures['brand'].astype(str), testFeatures['company'].astype(str), testFeatures['category'].astype(str), testFeatures['offer'].astype(str), testFeatures['chain'].astype(str), testFeatures['market'].astype(str)], ['id','brand', 'company', 'category', 'offer', 'chain', 'market'])
testFeaturesUpdated.add_columns(temp)


# adding more features

# In[ ]:

hasNeverBoughtCompany = -1*(trainFeaturesUpdated['numCompany'] == 0)
hasNeverBoughtBrand = -1*(trainFeaturesUpdated['numBrand'] == 0)
hasNeverBoughtCategory = -1*(trainFeaturesUpdated['numCategory'] == 0)

hasBoughtCompanyBrand = 1*((trainFeaturesUpdated['numCompany'] > 0)*(trainFeaturesUpdated['numBrand'] > 0))
hasBoughtCompanyCategory = 1*((trainFeaturesUpdated['numCompany'] > 0)*(trainFeaturesUpdated['numCategory'] > 0))
hasBoughtCategoryBrand = 1*((trainFeaturesUpdated['numCategory'] > 0)*(trainFeaturesUpdated['numBrand'] > 0))

trainFeaturesUpdated.add_columns([hasNeverBoughtCompany,hasNeverBoughtBrand,hasNeverBoughtCategory],['hasNeverBoughtCompany','hasNeverBoughtBrand','hasNeverBoughtCategory'])
trainFeaturesUpdated.add_columns([hasBoughtCompanyBrand,hasBoughtCategoryBrand,hasBoughtCompanyCategory],['hasBoughtCompanyBrand','hasBoughtCategoryBrand','hasBoughtCompanyCategory'])

hasBoughtCompanyBrand = 1*((testFeaturesUpdated['numCompany'] > 0)*(testFeaturesUpdated['numBrand'] > 0))
hasBoughtCompanyCategory = 1*((testFeaturesUpdated['numCompany'] > 0)*(testFeaturesUpdated['numCategory'] > 0))
hasBoughtCategoryBrand = 1*((testFeaturesUpdated['numCategory'] > 0)*(testFeaturesUpdated['numBrand'] > 0))
testFeaturesUpdated.add_columns([hasBoughtCompanyBrand,hasBoughtCategoryBrand,hasBoughtCompanyCategory],['hasBoughtCompanyBrand','hasBoughtCategoryBrand','hasBoughtCompanyCategory'])

hasBoughtCompanyBrandCategory = 1*((trainFeaturesUpdated['numCompany'] > 0)*(trainFeaturesUpdated['numBrand'] > 0)*(trainFeaturesUpdated['numCategory'] > 0))
trainFeaturesUpdated.add_column(hasBoughtCompanyBrandCategory,'hasBoughtCompanyBrandCategory')

hasBoughtCompanyBrandCategory = 1*((testFeaturesUpdated['numCompany'] > 0)*(testFeaturesUpdated['numBrand'] > 0)*(testFeaturesUpdated['numCategory'] > 0))
testFeaturesUpdated.add_column(hasBoughtCompanyBrandCategory,'hasBoughtCompanyBrandCategory')

hasNeverBoughtCompany = -1*((testFeaturesChange['numCompany'] == 0))
hasNeverBoughtCategory = -1*((testFeaturesChange['numCategory'] == 0))
hasNeverBoughtBrand = -1*((testFeaturesChange['numBrand'] == 0))
testCheck1.add_columns([hasNeverBoughtCompany,hasNeverBoughtCategory, hasNeverBoughtBrand],['hasNeverBoughtCompany','hasNeverBoughtCategory','hasNeverBoughtBrand'])


# In[ ]:

**************************


# And some more features 

# In[ ]:

itemsBrandDate = items[:]
itemsCategoryDate = items[:]
itemsCompanyDate = items[:]
itemsBrandDate.remove_columns(['chain', 'dept', 'productsize','productmeasure','category','company'])
itemsCategoryDate.remove_columns(['chain', 'dept', 'productsize','productmeasure','brand','company'])
itemsCompanyDate.remove_columns(['chain', 'dept', 'productsize','productmeasure','category','brand'])

itemsBrandDate.remove_columns(['purchasequantity','purchaseamount'])
itemsCategoryDate.remove_columns(['purchasequantity','purchaseamount'])
itemsCompanyDate.remove_columns(['purchasequantity','purchaseamount'])



# In[ ]:

tic = time.time()
dictStore = {}
date_format = "%Y-%m-%d"
pattern = "%Y-%m-%d"
lIdT = []
l = []
with open("trainFeatures.csv") as f:
    for i in xrange(1):
        f.next()
    for e, line in enumerate( f ):
        dateNowString = line.split(",")[6]
        dateNow = line.split(",")[6][1:len(dateNowString)-1]
        ep = datetime.strptime(dateNow, date_format)
        l = [int(line.split(",")[7]), int(line.split(",")[9]), int(line.split(",")[11]), ep]
        dictStore[int(line.split(",")[0])] = l
        #dictStore[line.split(",")[0]] = l
        lIdT.append(int(line.split(",")[0]))
        toc = time.time()
print (toc - tic)
print len(dictStore)
print len(lIdT)


# In[ ]:

tic = time.time()
i = 0
date_format = "%Y-%m-%d"
listOfValuesBrand = []
listOfValuesCompany = []
listOfValuesCategory = []
keyError = []
for x in lIdT:
    #print type(x)
    i = i + 1
    presentValues = [0,0,0,0,0,0,0]#15,30,60,90,120,150,180 in reverse
    if((i % 50) == 0):
        print (time.time()-tic), i
    try:
        xlist = listofIndices[x]
        particularIdBrandDate = itemsBrandDate[xlist[0]:xlist[1]]#particular id is a sframe
        particularIdCompanyDate = itemsCompanyDate[xlist[0]:xlist[1]]
        particularIdCategoryDate = itemsCategoryDate[xlist[0]:xlist[1]]
        listDictStore = dictStore[x]
        dateNow = listDictStore[3]
        
        particularIdCategoryDate = particularIdCategoryDate.filter_by([listDictStore[0]], 'category')    
        particularIdCategoryDate = list(particularIdCategoryDate['date'])
        if(len(particularIdCategoryDate)!= 0):
            for eachDate in reversed(particularIdCategoryDate):
                eachDateNew = datetime.strptime(eachDate, date_format)
                delta = dateNow - eachDateNew
                daysDiff = delta.days
                if(daysDiff < 15):
                    presentValues = [x+1 for x in presentValues]
                elif(daysDiff < 30):
                    presentValues[:6] = [x+1 for x in presentValues[:6]]
                elif(daysDiff < 60):
                    presentValues[:5] = [x+1 for x in presentValues[:5]]
                elif(daysDiff < 90):
                    presentValues[:4] = [x+1 for x in presentValues[:4]]
                elif(daysDiff < 120):
                    presentValues[:3] = [x+1 for x in presentValues[:3]]
                elif(daysDiff < 150):
                    presentValues[:2] = [x+1 for x in presentValues[:2]]
                elif(daysDiff < 180):
                    presentValues[:1] = [x+1 for x in presentValues[:1]]
                else:
                    break
                
        listOfValuesCategory.append(presentValues)

        presentValues = [0,0,0,0,0,0,0]
        particularIdCompanyDate = particularIdCompanyDate.filter_by([listDictStore[1]], 'company')    
        particularIdCompanyDate = list(particularIdCompanyDate['date'])
        if(len(particularIdCompanyDate)!= 0):
            for eachDate in reversed(particularIdCompanyDate):
                eachDateNew = datetime.strptime(eachDate, date_format)
                delta = dateNow - eachDateNew
                daysDiff = delta.days
                if(daysDiff < 15):
                    presentValues = [x+1 for x in presentValues]
                elif(daysDiff < 30):
                    presentValues[:6] = [x+1 for x in presentValues[:6]]
                elif(daysDiff < 60):
                    presentValues[:5] = [x+1 for x in presentValues[:5]]
                elif(daysDiff < 90):
                    presentValues[:4] = [x+1 for x in presentValues[:4]]
                elif(daysDiff < 120):
                    presentValues[:3] = [x+1 for x in presentValues[:3]]
                elif(daysDiff < 150):
                    presentValues[:2] = [x+1 for x in presentValues[:2]]
                elif(daysDiff < 180):
                    presentValues[:1] = [x+1 for x in presentValues[:1]]
                else:
                    break
                
        listOfValuesCompany.append(presentValues)

        presentValues = [0,0,0,0,0,0,0]
        particularIdBrandDate = particularIdBrandDate.filter_by([listDictStore[2]], 'brand')    
        particularIdBrandDate = list(particularIdBrandDate['date'])
        if(len(particularIdBrandDate)!= 0):
            for eachDate in reversed(particularIdBrandDate):
                eachDateNew = datetime.strptime(eachDate, date_format)
                delta = dateNow - eachDateNew
                daysDiff = delta.days
                if(daysDiff < 15):
                    presentValues = [x+1 for x in presentValues]
                elif(daysDiff < 30):
                    presentValues[:6] = [x+1 for x in presentValues[:6]]
                elif(daysDiff < 60):
                    presentValues[:5] = [x+1 for x in presentValues[:5]]
                elif(daysDiff < 90):
                    presentValues[:4] = [x+1 for x in presentValues[:4]]
                elif(daysDiff < 120):
                    presentValues[:3] = [x+1 for x in presentValues[:3]]
                elif(daysDiff < 150):
                    presentValues[:2] = [x+1 for x in presentValues[:2]]
                elif(daysDiff < 180):
                    presentValues[:1] = [x+1 for x in presentValues[:1]]
                else:
                    break
                
        listOfValuesBrand.append(presentValues)

    except:
        keyError.append(x)
        #print x

toc = time.time()
print (toc-tic)


# In[ ]:

**********************************


# Adding few more features, Making a model and predicting

# In[ ]:

tic = time.time()
i = 0
numberOfTransactions = []
keyError = []
for x in lIdT:
    try:
        xNum = listofIndices[x]
        numberOfTransactions.append(xNum)
    except:
        numberOfTransactions.append(100000000)

toc = time.time()
print (toc-tic)

trainFeaturesChange = gl.load_sframe('./trainFeaturesUpdated')
testFeaturesChange = gl.load_sframe('./testFeaturesUpdated')
trainFeaturesChange.remove_column('offerdate')
testFeaturesChange.remove_column('offerdate')

numberOfTransactions = SArray(numberOfTransactions)
trainFeaturesChange.add_column(numberOfTransactions, 'numberOfTransactions')
companyLoyalty = trainFeaturesChange['numCompany']/trainFeaturesChange['numberOfTransactions'] 
trainFeaturesChange.add_column(companyLoyalty, 'companyLoyalty')
categoryLoyalty = trainFeaturesChange['numCategory']/trainFeaturesChange['numberOfTransactions'] 
trainFeaturesChange.add_column(categoryLoyalty, 'categoryLoyalty')

tic = time.time()
i = 0
numberOfTransactions = []
keyError = []
for x in lIdTTest:
    try:
        xNum = listofIndices[x]
        numberOfTransactions.append(xNum)
    except:
        numberOfTransactions.append(100000000)

toc = time.time()
print (toc-tic)

brandLoyalty = testFeaturesChange['numBrand']/testFeaturesChange['numberOfTransactions'] 
testFeaturesChange.add_column(brandLoyalty, 'brandLoyalty')
companyLoyalty = testFeaturesChange['numCompany']/testFeaturesChange['numberOfTransactions'] 
testFeaturesChange.add_column(companyLoyalty, 'companyLoyalty')
categoryLoyalty = testFeaturesChange['numCategory']/testFeaturesChange['numberOfTransactions'] 
testFeaturesChange.add_column(categoryLoyalty, 'categoryLoyalty')

trainCheck = trainFeaturesChange[(trainFeaturesChange['amountCategory'] <= 305) & (trainFeaturesChange['amountCategory'] >= 0)]
trainCheck = trainCheck[(trainCheck['amountCompany'] <= 380) & (trainCheck['amountCompany'] >= 0)]
trainCheck = trainCheck[(trainCheck['amountBrand'] <= 210) & (trainCheck['amountBrand'] >= 0)]
print len(trainCheck)

trainCheck1 = trainCheck[:]
hasNotBoughtAny180 = -1*((trainCheck['company180'] == 0)*(trainCheck['category180'] == 0)*(trainCheck['brand180'] == 0))
hasBoughtAll15 = 1*((trainCheck['company15'] > 0)*(trainCheck['category15'] > 0)*(trainCheck['brand15'] > 0))
trainCheck1.add_columns([hasNotBoughtAny180,hasBoughtAll15],['hasNotBoughtAny180','hasBoughtAll15'])
testCheck1 = testFeaturesChange[:]
hasNotBoughtAny180 = -1*((testFeaturesChange['company180'] == 0)*(testFeaturesChange['category180'] == 0)*(testFeaturesChange['brand180'] == 0))
hasBoughtAll15 = 1*((testFeaturesChange['company15'] > 0)*(testFeaturesChange['category15'] > 0)*(testFeaturesChange['brand15'] > 0))
testCheck1.add_columns([hasNotBoughtAny180,hasBoughtAll15],['hasNotBoughtAny180','hasBoughtAll15'])



# In[ ]:

***************************************


# -------------------Random Forest Classifier--------------------

# In[ ]:

model= gl.random_forest_classifier.create(trainCheck1,target = 'repeater', verbose = True, class_weights = 'auto')
predictions = model.classify(testCheck1)
upload = testCheck1['id', 'numBrand']
upload.remove_column('numBrand')
upload.add_column(predictions['probability'], 'repeatProbability')
upload.export_csv('submit.csv', delimiter = ',', line_terminator = '\n', header = True, quote_level=2, quote_char = '"', na_rep = 'NA', file_header = '', file_footer = '', line_prefix = '', _no_prefix_on_first_value=False)

