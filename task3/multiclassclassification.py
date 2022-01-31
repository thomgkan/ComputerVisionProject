import os
import cv2 as cv
import numpy as np
import json

train_folders = ['imagedb']

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc / np.sum(bow_desc)

#knn algorithm
def knn_classification(query_img_path, bow_descs, img_paths, k):
    query_img = cv.imread(query_img_path)
    # query image features
    desc = extract_local_features(query_img_path)
    bow_desc = encode_bovw_descriptor(desc, vocabulary)
    cv.namedWindow('query', cv.WINDOW_NORMAL)
    cv.imshow('query', query_img)

    s = np.sum((bow_desc - bow_descs) ** 2, axis=1)
    distances = np.sqrt(s)
    all_distances = np.argsort(distances)

    retrieved_ids = all_distances[0:k]
    folderpath=[]
    for id in retrieved_ids.tolist():
        pathid = os.path.split(img_paths[id])
        pathsplit = os.path.split(pathid[0])
        folderpath.append(pathsplit[1])
    fpath = {}
    for f in folderpath:
        fpath[f] = folderpath.count(f)
    sortedpath = sorted(fpath, key=fpath.get, reverse=True)
    cv.namedWindow('results', cv.WINDOW_NORMAL)
    return os.path.join(train_folders[0], sortedpath[0])

def svm_classification (query_img_path, bow_descs, img_paths, kernel_type,):
    query_img = cv.imread(query_img_path)
    # query image features
    desc = extract_local_features(query_img_path)
    bow_desc = encode_bovw_descriptor(desc, vocabulary)
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(kernel_type)
    svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
    finalpath = 0
    for folder in os.listdir(train_folders[0]):
        #print(folder)
        print(folder)
        labels = np.array([folder in a for a in img_paths], np.int32)
        svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response[1] < 0:
            finalpath = os.path.join(train_folders[0], folder)
            break
        else:
            continue

    cv.namedWindow('query', cv.WINDOW_NORMAL)
    cv.imshow('query', query_img)
    if finalpath == 0:
        results = 'change the parameters, match not found'
        print(results)
    else:
        results = finalpath
    return results

# print('Extracting features...')
# train_descs = np.zeros((0, 128))
# for mainfolder in train_folders:
#     folders = os.listdir(mainfolder)
#     for subfolders in folders:
#         path = os.path.join(mainfolder, subfolders)
#         for files in os.listdir(path):
#             path = os.path.join(mainfolder, subfolders, files)
#             desc = extract_local_features(path)
#             if desc is None: # For Thumbs.db
#                 continue
#             train_descs = np.concatenate((train_descs, desc), axis=0)
# #
# # # Create vocabulary
# print('Creating vocabulary...')
# term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
# loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), 100, None, term_crit, 1, 0)
# np.save('vocabulary3.npy', vocabulary)

# Load vocabulary
vocabulary = np.load('vocabulary.npy')

# #Create Index
# print('Creating index...')
# img_paths = []
# train_descs = np.zeros((0, 128))
# bow_descs = np.zeros((0, vocabulary.shape[0]))
# for mainfolder in train_folders:
#     folders = os.listdir(mainfolder)
#     for subfolders in folders:
#         path = os.path.join(mainfolder, subfolders)
#         for files in os.listdir(path):
#             path = os.path.join(mainfolder, subfolders, files)
#             desc = extract_local_features(path)
#             if desc is None:
#                 continue
#             bow_desc = encode_bovw_descriptor(desc, vocabulary)
#
#             img_paths.append(path)
#             bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
#
# np.save('index3.npy', bow_descs)
# with open('index_paths3.txt', mode='w+') as file:
#     json.dump(img_paths, file)

# Load Index
bow_descs = np.load('index.npy')
with open('index_paths.txt', mode='r') as file:
    img_paths = json.load(file)


# search image
#query_img_path = 'imagedb_test/00012/00263_00000.ppm'


test_folder = ['imagedb_test']

#K-NN μαζί με την επανάληψη για την αξιολόγηση
for i in range(4):
    if i==0:
        k=5
    elif i==1:
        k=10
    elif i==2:
        k=25
    else:
        k=50
    for f in test_folder:
        success = {}
        totalsuccess = 0
        totaltest = 0
        for f2 in os.listdir(f):
            pathquery=os.path.join(f,f2)

            samples = 0
            for imgfiles in os.listdir(pathquery):
                query_img_path = os.path.join(pathquery,imgfiles)
                #k-NN
                finalpath = knn_classification(query_img_path, bow_descs, img_paths, k)
                for files in os.listdir(finalpath):
                    similarimages = cv.imread(os.path.join(finalpath, files))
                    cv.imshow('results', similarimages)
                    cv.waitKey(0)
                if os.path.split(finalpath)[1] == os.path.split(pathquery)[1]:
                    samples += 1
            success[f2] = samples*100/len(os.listdir(pathquery))
            totalsuccess += samples
            totaltest += len(os.listdir(pathquery))
    print(success)
    print(100 * totalsuccess / totaltest)


# # SVM μαζί με την επανάληψη για την αξιολόγηση
for i in range(2):
    if i == 0 :
        kernel = cv.ml.SVM_RBF
    elif i == 1:
        kernel = cv.ml.SVM_CHI2

    for f in test_folder:
        success = {}
        totalsuccess = 0
        totaltest = 0
        for f2 in os.listdir(f):
            pathquery=os.path.join(f,f2)
            samples = 0
            for imgfiles in os.listdir(pathquery):
                query_img_path = os.path.join(pathquery,imgfiles)
                finalpath = svm_classification(query_img_path, bow_descs, img_paths, kernel)
                cv.namedWindow('results', cv.WINDOW_NORMAL)
                for files in os.listdir(finalpath):
                      cv.imshow('results', cv.imread(os.path.join(finalpath, files)))
                      cv.waitKey(0)
                if os.path.split(finalpath)[1] == os.path.split(pathquery)[1]:
                    samples += 1
            success[f2] = samples*100/len(os.listdir(pathquery))
            totalsuccess += samples
            totaltest += len(os.listdir(pathquery))
    print(kernel)
    print(success)
    print(100 * totalsuccess / totaltest)
