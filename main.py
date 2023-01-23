import sqlalchemy
import pymysql
import sqlalchemy as db
import pandas as pd


class DataAdditionSQL:
    def __init__(self):
        engine = db.create_engine("mysql+pymysql://root:password@localhost/WrittenAssignment")
        # get connection object
        self.meta_data = db.MetaData()
        self.connection = engine.connect()


def main():
    # get sqlalchemy and pymysql used libraries version
    print("sqlalchemy: {}".format(sqlalchemy.__version__))
    print("pymysql: {}".format(pymysql.__version__))
    # url = 'mysql://%s:%s@%s' % ('root', 'password', 'localhost')
    # engine = db.create_engine(url)  # connect to server
    #
    # create_str = "CREATE DATABASE IF NOT EXISTS %s ;" % 'movie'
    # engine.execute(create_str)
    # engine.execute("USE location;")
    # db.create_all()
    # db.session.commit()
    # get engine object using pymysql driver for mysql
    engine = db.create_engine("mysql+pymysql://root:password@localhost/movie")
    # get connection object
    meta_data = db.MetaData()
    connection = engine.connect()
    # get meta data object
    actor_table = db.Table("actor", meta_data, autoload=True,
                           autoload_with=engine)
    # set the insert statement
    sql_query = db.insert(actor_table)
    # set data list
    data_list = [{"first_name": "John", "last_name": "Smith", "age": 50,
                  "date_of_birth": "1969-04-05", "active": True},
                 {"first_name": "Brian", "last_name": "Morgan", "age": 38,
                  "date_of_birth": "1981-02-11", "active": True},
                 {"first_name": "David", "last_name": "White", "age": 77,
                  "date_of_birth": "1942-06-30", "active": False}]
    # execute the insert statement
    result = connection.execute(sql_query, data_list)


if __name__ == '__main__':
    main()
