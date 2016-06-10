from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName("main")
sc = SparkContext(conf=conf)

# constants definition
FIELD_YEAR = 0
FIELD_MONTH = 1
FIELD_DAY_OF_MONTH = 2
FIELD_DAY_OF_WEEK = 3
FIELD_DEP_TIME = 4
FIELD_CRS_DEP_TIME = 5
FIELD_ARR_TIME = 6
FIELD_CRS_ARR_TIME = 7
FIELD_UNIQUE_CARRIER = 8
FIELD_FLIGHT_NUM = 9
FIELD_TAIL_NUM = 10
FIELD_ACTUAL_ELAPSED_TIME = 11
FIELD_CRS_ELAPSED_TIME = 12
FIELD_AIR_TIME = 13
FIELD_ARR_DELAY = 14
FIELD_DEP_DELAY = 15
FIELD_ORIGIN = 16
FIELD_DEST = 17
FIELD_DISTANCE = 18
FIELD_TAXI_IN = 19
FIELD_TAXI_OUT = 20
FIELD_CANCELLED = 21
FIELD_CANCELLATION_CODE = 22
FIELD_DIVERTED = 23
FIELD_CARRIER_DELAY = 24
FIELD_WEATHER_DELAY = 25
FIELD_NAS_DELAY = 26
FIELD_SECURITY_DELAY = 27
FIELD_LATE_AIRCRAFT_DELAY = 28

def split(line):
    return line.split(',')

def parseTrain(parts):
    p = parts
    label = p[FIELD_CANCELLED]
    features = [
        p[FIELD_FLIGHT_NUM],
        # p[FIELD_DISTANCE] if p[FIELD_DISTANCE] != 'NA' else 0,
    ]
    return LabeledPoint(label, features)

def parseTest(parts):
    p = parts
    label = p[FIELD_CANCELLED]
    features = [
        p[FIELD_FLIGHT_NUM],
        # p[FIELD_DISTANCE] if p[FIELD_DISTANCE] != 'NA' else 0,
    ]
    return LabeledPoint(label, features)

def toCSVLine(data):
    return ','.join(str(d) for d in data)

def main():
    # prepare training data
    # RDDTrainData = sc.textFile('2007_100.csv')
    RDDTrainData = sc.textFile(','.join([
        # '1987.csv',
        # '1988.csv',
        # '1989.csv',
        # '1990.csv',
        # '1991.csv',
        # '1992.csv',
        # '1993.csv',
        # '1994.csv',
        # '1995.csv',
        # '1996.csv',
        # '1997.csv',
        # '1998.csv',
        # '1999.csv',
        # '2000.csv',
        # '2001.csv',
        # '2002.csv',
        # '2003.csv',
        # '2004.csv',
        # '2005.csv',
        # '2006.csv',
        '2007.csv',
    ]))
    RDDTrainHeader = RDDTrainData.take(1)[0]
    trainData = RDDTrainData.filter(lambda line: line != RDDTrainHeader)\
                            .map(split)\
                            .map(parseTrain)

    # prepare testing data
    RDDTestData = sc.textFile('2008.csv')
    RDDTestHeader = RDDTestData.take(1)[0]
    testData = RDDTestData.filter(lambda line: line != RDDTestHeader)\
                          .map(split)\
                          .map(parseTest)

    # do prediction

    # SVM
    model = SVMWithSGD.train(trainData, iterations=100)

    # Logistic Regression
    # model = LogisticRegressionWithLBFGS.train(trainData)

    predictionData = testData.map(lambda d:
        (int(d.label), model.predict(d.features))
    )

    # evaluate error rate
    errorCount = predictionData.filter(lambda d: int(d[0]) != int(d[1])).count()
    totalCount = predictionData.count()
    print 'error rate =', errorCount, '/', totalCount, '=', float(errorCount) / float(totalCount)

main()
