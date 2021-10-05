# +
import sys
from scipy.spatial import distance as dist
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import numpy as np

class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False

class CentroidTracker:
    def __init__(self, maxDisappeared=5, maxDistance=50,minAppeared=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()        
        self.maxDisappeared = maxDisappeared

        self.appeared = OrderedDict()
        self.minAppeared = minAppeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.appeared[objectID]

    def update(self, rects):

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        for objectID in list(self.appeared.keys()):            
            self.appeared[objectID] += 1

        # pdb.set_trace()
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):            
                self.register(inputCentroids[i])

        else:

            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                    

            else:
                for col in unusedCols:
                    objectID = objectIDs[row]
                    self.register(inputCentroids[col])

        return self.objects



class REID_tracker:
    def __init__(self, maxDisappeared=5, minimum_similarity=.5,minAppeared=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()        
        self.maxDisappeared = maxDisappeared

        self.appeared = OrderedDict()
        self.minAppeared = minAppeared

        self.minimum_similarity = minimum_similarity

    def register(self, centroid,embedding):
        self.objects[self.nextObjectID] = (centroid,embedding)
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.appeared[objectID]

    def update(self, rects,embeddings):


        if len(embeddings) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        for objectID in list(self.appeared.keys()):            
            self.appeared[objectID] += 1

        # pdb.set_trace()
        
        inputEmbeddings = np.zeros((len(embeddings), 512), dtype="int")
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY),embedding) in enumerate(zip(rects,embeddings)):

            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputEmbeddings[i] = embedding


        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):            
                self.register(inputCentroids[i],inputEmbeddings[i])

        else:

            objectIDs = list(self.objects.keys())
            objectCentroids = [i[0] for i in list(self.objects.values())]
            objectEmbeddings = [i[1] for i in list(self.objects.values())]


            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.max(axis=1).argsort()
            cols = D.argmax(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] < self.minimum_similarity:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                    

            else:
                for col in unusedCols:
                    objectID = objectIDs[row]
                    self.register(inputCentroids[col])

        return self.objects
# +
# import pickle

# embedding = pickle.load(open("sequential_embedding.pickle","rb"))
# # -

# from sklearn.metrics import pairwise

# pairwise.cosine_similarity(embedding[1500][0],embedding[1][0])

# embedding[101]

# dist.cosine(embedding[0][0],embedding[1][0])

# embedding[0][0]

# dist.
