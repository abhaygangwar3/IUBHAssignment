import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import sqlalchemy as db
import warnings
import unittest
import sys


class WrittenAssignment:
    '''This class will be used for all the main tasks present in the written assignment
       All the datasets provided in the assignment has been kept in "Datasets" directory inside the project'''

    def __init__(self, file_path_train='train.csv', file_path_test='test.csv', file_path_ideal_funct='ideal.csv'):
        home_dir = sys.path[1]
        file_path_train = home_dir + '\\Datasets\\' + file_path_train
        file_path_test = home_dir + '\\Datasets\\' + file_path_test
        file_path_ideal_funct = home_dir + '\\Datasets\\' + file_path_ideal_funct
        # Error handling
        try:
            self.trainDF = pd.read_csv(
                file_path_train)  # Using read_csv method of Pandas library to create data frame from the provided path
            self.testDF = pd.read_csv(file_path_test)
            self.idealDF = pd.read_csv(file_path_ideal_funct)
        except IOError:
            print("There is some error with the file or directory. Kindly recheck the name.")

    # Purpose of this method is to return the square deviation between two provided pandas series
    def square_deviation(self, train_series, ideal_series):
        res_temp = (ideal_series - train_series) ** 2
        res = np.mean(res_temp)  # We are using numpy library to find the mean of the Square deviation between series
        return res

    # function to find minimum and maximum position in list
    def minimum(self, a):

        # inbuilt function to find the position of minimum
        minval = min(a)
        minpos = a.index(minval)

        # returning the position and value
        return minval, minpos + 1

    # Actual function to map the ideal fucntion to the training data set
    def idealFunctionFinder(self):
        newDF_ideal = self.idealDF.drop('x', axis=1)
        newDF_train = self.trainDF.drop('x', axis=1)
        listIdealFn = newDF_ideal.columns.to_list()
        listTrainFn = newDF_train.columns.to_list()
        res = []
        self.idealFunction = []  # A blank list which will be used to store the position of the ideal functions
        for i in listTrainFn:
            col = []
            for j in listIdealFn:
                r = self.square_deviation(newDF_train[i], newDF_ideal[j])
                col.append(r)
            minval, minpos = self.minimum(col)
            res.append(col)
            # print(f"Least squared deviation for index y{minpos} is {minval}")
            self.idealFunction.append(minpos)
        idealFunction = list(map(str, self.idealFunction))
        idealFunctCol = []
        for i in idealFunction:
            idealFunctCol.append('y' + i)
        self.dictTrainIdeal = {newDF_train.columns.values.tolist()[i]: idealFunctCol[i] for i in
                               range(
                                   len(idealFunctCol))}  # This creates a dictionary which contains the mapping between the training data set and corresponding ideal function

    # Function for visualization of the dataset.
    def visualisationTrainingSet(self):
        # Creating the scatter plot between the training data set and the corresponding ideal function to visualise the similarity
        p1 = figure(title='Training set 1 vs corresponding ideal function', x_axis_label='x', y_axis_label='y',
                    width=350, height=350)
        p1.scatter(self.trainDF['x'], self.trainDF['y1'], line_width=2, line_color='green')
        p1.scatter(self.trainDF['x'], self.idealDF[self.dictTrainIdeal.get('y1')], line_width=2, line_color='blue')

        p2 = figure(title='Training set 2 vs corresponding ideal function', x_axis_label='x', y_axis_label='y',
                    width=350, height=350)
        p2.scatter(self.trainDF['x'], self.trainDF['y2'], line_width=2, line_color='green')
        p2.scatter(self.trainDF['x'], self.idealDF[self.dictTrainIdeal.get('y2')], line_width=2, line_color='blue')

        p3 = figure(title='Training set 3 vs corresponding ideal function', x_axis_label='x', y_axis_label='y',
                    width=350, height=350)
        p3.scatter(self.trainDF['x'], self.trainDF['y3'], line_width=2, line_color='green')
        p3.scatter(self.trainDF['x'], self.idealDF[self.dictTrainIdeal.get('y3')], line_width=2, line_color='blue')

        p4 = figure(title='Training set 4 vs corresponding ideal function', x_axis_label='x', y_axis_label='y',
                    width=350, height=350)
        p4.scatter(self.trainDF['x'], self.trainDF['y4'], line_width=2, line_color='green')
        p4.scatter(self.trainDF['x'], self.idealDF[self.dictTrainIdeal.get('y4')], line_width=2, line_color='blue')

        p = gridplot([[p1, p2], [p3, p4]])
        show(p)

    def visualisationTestingSet(self):
        # This is to visualise the training data set with the test data set so that it becomes easy to view the difference
        p = figure(title="Training Dataset vs Testing Dataset", x_axis_label='x', y_axis_label='y')
        p.scatter(self.trainDF['x'], self.trainDF['y4'], line_width=2, color='blue', legend_label='Training dataset 4')
        p.scatter(self.testDF['x'], self.testDF['y'], line_width=2, color='red', legend_label='Testing dataset')
        p.scatter(self.trainDF['x'], self.trainDF['y1'], line_width=2, color='green', legend_label='Training dataset 1')
        p.scatter(self.trainDF['x'], self.trainDF['y2'], line_width=2, color='black', legend_label='Training dataset 2')
        p.scatter(self.trainDF['x'], self.trainDF['y3'], line_width=2, color='yellow',
                  legend_label='Training dataset 3')
        show(p)

    # Below function is used to map the test data set with the ideal function (if it exists post application of the criteria) and preparing data frame accordingly

    def testDataFiltering(self):
        indexTest = []
        for x in range(0, len(self.testDF['x'])):
            indexTest.append(self.trainDF['x'].to_list().index(
                self.testDF['x'][
                    x]))  # To find the index of the x values present in test set mapped with the training set
        maxDevTrainIdeal = []
        for i, j in self.dictTrainIdeal.items():
            maxDevTrainIdeal.append(max(self.idealDF[j] - self.trainDF[i]))
        testSetToIdealFunct = []
        deviationWithIdealFunct = []
        counter = 0
        for i in indexTest:
            flag = ""
            deviation = 0.0
            testtemp2 = float(self.testDF[counter:counter + 1]['y'])
            count = 0
            for j in self.dictTrainIdeal.values():
                idealtemp1 = float(self.idealDF[i:i + 1][j])
                dif = abs(idealtemp1 - testtemp2)
                cond = np.sqrt(2) * maxDevTrainIdeal[count]
                # print(f"{idealtemp1} - {testtemp2} = {dif} and max train diff is {cond}")
                if (
                        dif <= cond):  # 'dif' contains the absolute difference between the ideal function value and test data set and 'cond' contains the value as per the provided criteria
                    flag = j
                    deviation = dif
                    break
                count += 1
            counter += 1
            if flag != "":
                testSetToIdealFunct.append(flag)
                deviationWithIdealFunct.append(str(deviation))
            else:
                testSetToIdealFunct.append('NA')
                deviationWithIdealFunct.append('NA')

        # Creating final test data set along with the specified format

        self.testDF['Delta Y (test funct)'] = deviationWithIdealFunct
        self.testDF['No. of ideal funct'] = testSetToIdealFunct
        self.testDF = self.testDF.rename(columns={'x': 'X (test func)',
                                                  'y': 'Y (test func)'})  # Renaming the test data frame as per the standard present in the written assignment


class SQLWorkBench(WrittenAssignment):
    '''This class is used as child class to the already created class WrittenAssignment so that the already existing data is used in this class
    It contains functions which are needed for the second part of the assignment where we are inserting the data in the MySQL DB in a speciofied format'''

    def __init__(self, testDFMod, DBName="Assignment"):

        self.engine = db.create_engine(f"sqlite:///{DBName}.db")  # Create SQLite Database with 'DBName'

        # Use below code to use MySQL database for data insertion

        # # get engine object using pymysql driver for mysql
        # self.engine = db.create_engine("mysql+pymysql://root:password@localhost/IUBHAssignment",pool_pre_ping=True)
        # # get connection object
        # self.connection = self.engine.connect()
        # # get meta data object
        # self.meta_data = db.MetaData()

        WrittenAssignment.__init__(self)
        self.testDFMod = testDFMod

    def renamingColumns(self):
        # It is used for renaming the column names as per the assignment before inserting into MySQL DB
        dictColumnNameIdeal = {
            self.idealDF.columns.values.tolist()[i]: self.idealDF.columns.values.tolist()[i].upper() + ' (ideal func)'
            for i in
            range(len(self.idealDF.columns))}
        self.idealDFMod = self.idealDF.rename(columns=dictColumnNameIdeal)
        self.idealDFMod = self.idealDFMod.rename(columns={'X (ideal func)': 'X'})

        dictColumnNameTrain = {
            self.trainDF.columns.values.tolist()[i]: self.trainDF.columns.values.tolist()[
                                                         i].upper() + ' (training func)' for i in
            range(len(self.trainDF.columns))}
        self.trainDFMod = self.trainDF.rename(columns=dictColumnNameTrain)
        self.trainDFMod = self.trainDFMod.rename(columns={'X (training func)': 'X'})

    # This is used for creating table and inserting data using the dataframe in the already existing database.
    def bulkinsertData(self, name_of_table):
        name = name_of_table.capitalize()
        if (name == "Ideal functions"):
            self.idealDFMod.to_sql(name, self.engine, if_exists='replace', index=False)
        elif (name == "Training data"):
            self.trainDFMod.to_sql(name, self.engine, if_exists='replace', index=False)
        else:
            self.testDFMod.to_sql(name, self.engine, if_exists='replace', index=False)


class MyException(Exception):
    '''User defined excpetion handling'''

    def __init___(self, exception_parameter, exception_message):
        super().__init__(self, exception_parameter, exception_message)


class UnitTestFileCheck(unittest.TestCase):

    def test_idealFunctionFinder(self):
        # To check the ideal function mapping with training data set and test data set
        try:
            WAObj = WrittenAssignment()  # Creating object of the class WrittenAssignment
            WAObj.idealFunctionFinder()
            WAObj.testDataFiltering()
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_SQLDBUpdate(self):
        # To check if the SQL process is getting completed or not
        try:
            SQL = SQLWorkBench()  # Creating object of the class SQLWorkBench
            SQL.renamingColumns()
            warnings.filterwarnings(action='ignore',
                                    category=UserWarning)  # This is used for ignoring a user warning which is not impacting the execution of the program
            counter = 0
            while (counter < 3):
                message = input(
                    "Provide the dataframe (Ideal Functions, Training Data, Test Data) you want to insert in the database as table: ")
                message = message.lower()
                try:
                    if message == "ideal functions" or message == "training data" or message == "test data":
                        SQL.bulkinsertData(message)  # This is to insert the required data in the SQL database
                        counter += 1
                    else:
                        raise MyException(message,
                                          "Unexpected input is received: {} .Please enter correct input".format(
                                              message))
                except MyException:
                    print("Incorrect value has been provided. Pleas try again.")
                    continue
            self.assertTrue(True)
        except:
            self.assertTrue(False)


if __name__ == '__main__':  # This is where the execution actually starts

    # unittest.main() # Un-comment to perform unit testing
    WAObj = WrittenAssignment()  # Creating object of the class WrittenAssignment
    WAObj.idealFunctionFinder()
    WAObj.visualisationTrainingSet()
    WAObj.visualisationTestingSet()
    WAObj.testDataFiltering()
    SQL = SQLWorkBench(WAObj.testDF)  # Creating object of the class SQLWorkBench
    SQL.renamingColumns()
    warnings.filterwarnings(action='ignore',
                            category=UserWarning)  # This is used for ignoring a user warning which is not impacting the execution of the program
    counter = 0
    while (counter < 3):
        message = input(
            "Provide the dataframe (Ideal Functions, Training Data, Test Data) you want to insert in the database as table: ")
        message = message.lower()
        try:
            if message == "ideal functions" or message == "training data" or message == "test data":
                SQL.bulkinsertData(message)  # This is to insert the required data in the SQL database
                counter += 1
            else:
                raise MyException(message,
                                  "Unexpected input is received: {} .Please enter correct input".format(message))
        except MyException:
            print("Incorrect value has been provided. Please try again.")
            continue
